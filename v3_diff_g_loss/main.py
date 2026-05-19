import os
import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import lightning as L
from lightning.pytorch.cli import LightningCLI

from diffusion import DiffusionBridge
from backbones.ncsnpp import NCSNpp
from datasets import DataModule
from utils import compute_metrics, save_image_pair, save_preds, save_eval_images


class SobelGradLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        kernel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("kernel_x", kernel_x)
        self.register_buffer("kernel_y", kernel_y)

    def forward(self, pred, target, reduction="mean"):
        gx_pred = F.conv2d(pred, self.kernel_x, padding=1)
        gy_pred = F.conv2d(pred, self.kernel_y, padding=1)
        gx_tgt = F.conv2d(target, self.kernel_x, padding=1)
        gy_tgt = F.conv2d(target, self.kernel_y, padding=1)
        grad_pred = torch.sqrt(gx_pred * gx_pred + gy_pred * gy_pred + 1e-6)
        grad_tgt = torch.sqrt(gx_tgt * gx_tgt + gy_tgt * gy_tgt + 1e-6)
        return F.l1_loss(grad_pred, grad_tgt, reduction=reduction)


class FinalPatchDiscriminator(torch.nn.Module):
    """Optional final-output discriminator; it never judges bridge intermediates."""

    def __init__(self, in_ch=1, base_ch=64):
        super().__init__()

        def block(cin, cout, stride):
            return torch.nn.Sequential(
                torch.nn.utils.spectral_norm(torch.nn.Conv2d(cin, cout, 4, stride=stride, padding=1)),
                torch.nn.LeakyReLU(0.2, inplace=True),
            )

        self.net = torch.nn.Sequential(
            block(in_ch, base_ch, 2),
            block(base_ch, base_ch * 2, 2),
            block(base_ch * 2, base_ch * 4, 2),
            torch.nn.utils.spectral_norm(torch.nn.Conv2d(base_ch * 4, 1, 3, padding=1)),
        )

    def forward(self, x):
        return self.net(x)


class BridgeRunner(L.LightningModule):
    def __init__(
        self,
        generator_params,
        diffusion_params,
        lr_g,
        lambda_rec_loss,
        optim_betas,
        eval_mask,
        eval_subject,
        rec_loss_reduction="sum",
        lambda_grad_loss=0.0,
        deep_supervision_gamma=0.5,
        lambda_adv_loss=0.0,
        lr_d=None,
        adv_start_epoch=50,
        d_update_interval=2,
        discriminator_params=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.lr_g = lr_g
        self.lr_d = lr_g if lr_d is None else lr_d
        self.lambda_rec_loss = lambda_rec_loss
        self.lambda_grad_loss = lambda_grad_loss
        self.lambda_adv_loss = float(lambda_adv_loss)
        self.optim_betas = optim_betas
        self.eval_mask = eval_mask
        self.eval_subject = eval_subject
        self.n_steps = diffusion_params["n_steps"]
        self.n_recursions = diffusion_params["n_recursions"]
        self.rec_loss_reduction = rec_loss_reduction
        self.deep_supervision_gamma = deep_supervision_gamma
        self.adv_start_epoch = int(adv_start_epoch)
        self.d_update_interval = int(d_update_interval)

        self.generator = NCSNpp(**generator_params)
        self.diffusion = DiffusionBridge(**diffusion_params)
        self.grad_loss_fn = SobelGradLoss()

        self.use_discriminator = self.lambda_adv_loss > 0.0
        if self.use_discriminator:
            discriminator_params = discriminator_params or {}
            self.discriminator = FinalPatchDiscriminator(**discriminator_params)
        else:
            self.discriminator = None

    def _resize_target_like(self, target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        if target.shape[-2:] == pred.shape[-2:]:
            return target
        return F.interpolate(target, size=pred.shape[-2:], mode="bilinear", align_corners=False)

    def _pixel_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target, reduction=self.rec_loss_reduction)

    @torch.no_grad()
    def _regression_accuracy(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        err = (pred - target).abs().mean()
        denom = target.abs().mean().clamp_min(1e-6)
        return (1.0 - err / denom).clamp(0.0, 1.0)

    def _build_layer_weights(self, n_levels: int, device: torch.device) -> torch.Tensor:
        raw = torch.tensor(
            [self.deep_supervision_gamma ** (n_levels - 1 - i) for i in range(n_levels)],
            dtype=torch.float32,
            device=device,
        )
        return raw / raw.sum()

    def _deep_supervision_loss(self, out_dict, gt: torch.Tensor, log_prefix: str = "train"):
        final_pred = out_dict["final"]
        aux_preds = out_dict.get("aux", [])
        preds = list(aux_preds) + [final_pred]
        weights = self._build_layer_weights(len(preds), gt.device)
        total_loss = gt.new_tensor(0.0)

        for i, (pred, weight) in enumerate(zip(preds, weights)):
            gt_i = self._resize_target_like(gt, pred)
            loss_i = self._pixel_loss(pred, gt_i)
            acc_i = self._regression_accuracy(pred.detach(), gt_i.detach())
            total_loss = total_loss + weight * loss_i
            self.log(f"{log_prefix}/layer_{i}_loss", loss_i, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log(f"{log_prefix}/layer_{i}_acc", acc_i, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log(f"{log_prefix}/layer_{i}_weight", weight, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        final_acc = self._regression_accuracy(final_pred.detach(), gt.detach())
        self.log(f"{log_prefix}/final_acc", final_acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return total_loss, final_pred

    def _lambda_map(self, t, y, x0_r):
        if hasattr(self.diffusion, "make_lambda_map"):
            return self.diffusion.make_lambda_map(t=t, y=y, x_r=x0_r, detach=True)
        return None

    def _recursive_predict(self, x_t, y, t, detach_xt: bool = False, return_aux_last: bool = False):
        x_in = x_t.detach() if detach_xt else x_t
        x0_r = torch.zeros_like(x_t)
        preds = []
        last_out = None

        for idx in range(self.n_recursions):
            need_aux = return_aux_last and idx == self.n_recursions - 1
            lambda_map = self._lambda_map(t, y, x0_r)
            out = self.generator(
                torch.cat((x_in, y), axis=1),
                t,
                x_r=x0_r,
                lambda_map=lambda_map,
                return_aux=need_aux,
            )
            if need_aux:
                x0_r = out["final"]
                last_out = out
            else:
                x0_r = out
            preds.append(x0_r)

        if return_aux_last:
            return preds, last_out
        return preds

    @staticmethod
    def _d_hinge_loss(real_logits, fake_logits):
        return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()

    @staticmethod
    def _g_hinge_loss(fake_logits):
        return -fake_logits.mean()

    def training_step(self, batch, batch_idx):
        x0, y, _ = batch
        opts = self.optimizers()
        scheds = self.lr_schedulers()
        if self.use_discriminator:
            optimizer_g, optimizer_d = opts
            scheduler_g = scheds[0] if isinstance(scheds, (list, tuple)) else scheds
        else:
            optimizer_g = opts
            optimizer_d = None
            scheduler_g = scheds

        optimizer_g.zero_grad()

        t = torch.randint(1, self.n_steps + 1, (x0.shape[0],), device=x0.device)
        x_t = self.diffusion.q_sample(t, x0, y, detach=False)
        _, final_out = self._recursive_predict(x_t, y, t, detach_xt=False, return_aux_last=True)
        x0_pred = final_out["final"]

        rec_loss, _ = self._deep_supervision_loss(final_out, x0, log_prefix="train")
        grad_loss = self.grad_loss_fn(x0_pred, x0, reduction=self.rec_loss_reduction) if self.lambda_grad_loss > 0 else x0.new_tensor(0.0)
        lambda_reg = self.diffusion.lambda_regularization(t, x0, y)

        adv_loss = x0.new_tensor(0.0)
        d_loss = x0.new_tensor(0.0)
        adv_active = self.use_discriminator and self.current_epoch >= self.adv_start_epoch

        if adv_active and (batch_idx % max(self.d_update_interval, 1) == 0):
            optimizer_d.zero_grad()
            real_logits = self.discriminator(x0.detach())
            fake_logits = self.discriminator(x0_pred.detach())
            d_loss = self._d_hinge_loss(real_logits, fake_logits)
            self.manual_backward(d_loss)
            optimizer_d.step()

        if adv_active:
            adv_loss = self._g_hinge_loss(self.discriminator(x0_pred))

        g_loss = self.lambda_rec_loss * rec_loss + self.lambda_grad_loss * grad_loss + lambda_reg + self.lambda_adv_loss * adv_loss
        self.manual_backward(g_loss)
        optimizer_g.step()
        if scheduler_g is not None:
            scheduler_g.step()

        self.log("g_loss/rec", rec_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("g_loss/grad", grad_loss, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("g_loss/lambda_reg", lambda_reg, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("g_loss/adv_final", adv_loss, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("g_loss/total", g_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        if self.use_discriminator:
            self.log("d_loss/final", d_loss, on_epoch=True, prog_bar=False, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        x0, y, _ = batch
        x0_pred = self.diffusion.sample_x0(y, self.generator)
        loss = F.mse_loss(x0_pred, x0)
        metrics = compute_metrics(x0, x0_pred)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_psnr", metrics["psnr_mean"].mean(), on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_ssim", metrics["ssim_mean"].mean(), on_epoch=True, prog_bar=True, sync_dist=True)
        if batch_idx == 0 and self.global_rank == 0:
            path = os.path.join(self.logger.log_dir, "val_samples", f"epoch_{self.current_epoch}.png")
            save_image_pair(x0, x0_pred, path)

    def on_test_start(self):
        self.test_samples = []
        self.mask = None
        self.subject_ids = None
        if self.eval_mask:
            self.mask = self.trainer.datamodule.test_dataset._load_data("mask")
        if self.eval_subject:
            self.subject_ids = self.trainer.datamodule.test_dataset.subject_ids

    def test_step(self, batch, batch_idx):
        x0, y, slice_idx = batch
        x0_pred = self.diffusion.sample_x0(y, self.generator)
        all_pred = self.all_gather(x0_pred)
        slice_indices = self.all_gather(slice_idx)
        if self.global_rank == 0:
            h, w = x0.shape[-2:]
            self.test_samples.extend(list(zip(slice_indices.flatten().tolist(), all_pred.reshape(-1, h, w).cpu().numpy())))

    def on_test_end(self):
        if self.global_rank == 0:
            self.test_samples.sort(key=lambda x: x[0])
            pred = np.array([x[1] for x in self.test_samples])
            slice_indices = np.array([x[0] for x in self.test_samples])
            _, locs = np.unique(slice_indices, return_index=True)
            pred = pred[locs]
            dataset = self.trainer.datamodule.test_dataset
            source = dataset.source
            target = dataset.target
            save_preds(pred, os.path.join(self.logger.log_dir, "test_samples", "pred.npy"))
            metrics = compute_metrics(
                gt_images=target,
                pred_images=pred,
                mask=self.mask,
                subject_ids=self.subject_ids,
                report_path=os.path.join(self.logger.log_dir, "test_samples", "report.txt"),
            )
            print(f"PSNR: {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f}")
            print(f"SSIM: {metrics['ssim_mean']:.2f} ± {metrics['ssim_std']:.2f}")
            indices = np.random.choice(len(dataset), 10)
            save_eval_images(
                source_images=source[indices],
                target_images=target[indices],
                pred_images=pred[indices],
                psnrs=metrics["psnrs"][indices],
                ssims=metrics["ssims"][indices],
                save_path=os.path.join(self.logger.log_dir, "test_samples"),
            )

    def configure_optimizers(self):
        optimizer_g = Adam(list(self.generator.parameters()) + list(self.diffusion.parameters()), lr=self.lr_g, betas=self.optim_betas)
        scheduler_g = CosineAnnealingLR(optimizer_g, T_max=self.trainer.max_epochs, eta_min=1e-5)
        if not self.use_discriminator:
            return [optimizer_g], [scheduler_g]
        optimizer_d = Adam(self.discriminator.parameters(), lr=self.lr_d, betas=self.optim_betas)
        return [optimizer_g, optimizer_d], [scheduler_g]


class _LightningCLI(LightningCLI):
    def instantiate_classes(self):
        if "test" in self.parser.args and "CSVLogger" in self.config.test.trainer.logger[0].class_path:
            exp_dir = os.path.dirname(os.path.dirname(self.config.test.ckpt_path))
            logger = self.config.test.trainer.logger[0]
            logger.init_args.save_dir = os.path.dirname(exp_dir)
            logger.init_args.name = os.path.basename(exp_dir)
            logger.init_args.version = "test"
        super().instantiate_classes()


def cli_main():
    _LightningCLI(BridgeRunner, DataModule, save_config_callback=None, parser_kwargs={"parser_mode": "omegaconf"})


if __name__ == "__main__":
    cli_main()
