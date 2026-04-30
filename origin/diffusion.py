import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import lightning as L


class AdaptiveBridgeLambda(nn.Module):
    """Content-aware adaptive bridge field with exact endpoint preservation.

    The mathematical role of lambda is unchanged: it only modifies the mean path
        m_t = (1 - lambda_t) * x0 + lambda_t * y.
    To keep the bridge interpretable and stable, the final lambda is constructed as
        lambda_t = (1 - focus_t) * lambda_linear + focus_t * lambda_adaptive,
    where focus_t is high only on anatomically valid, high-difference regions.
    As a result, background / low-difference areas stay close to the original linear
    bridge, while salient brain structures are allowed to deviate.
    """

    def __init__(
        self,
        channels: int = 1,
        degree: int = 5,
        enable_spatial_variation: bool = True,
        pos_hw=(256, 256),
        eps: float = 1e-4,
        focus_power: float = 1.5,
        focus_blur_kernel: int = 5,
        anatomy_threshold: float = 0.10,
        anatomy_slope: float = 10.0,
        focus_mix: float = 0.75,
    ):
        super().__init__()
        self.channels = channels
        self.enable_spatial_variation = enable_spatial_variation
        self.eps = float(eps)
        self.focus_power = float(focus_power)
        self.focus_blur_kernel = int(max(1, focus_blur_kernel))
        self.anatomy_threshold = float(anatomy_threshold)
        self.anatomy_slope = float(anatomy_slope)
        self.focus_mix = float(focus_mix)

        # Polynomial fallback branch.
        self.coeffs = nn.Parameter(torch.randn(channels, degree + 1) * 0.2)
        with torch.no_grad():
            for c in range(channels):
                self.coeffs.data[c, 1] = 1.0 + 0.2 * (c - channels // 2)
                self.coeffs.data[c, 0] = 0.0
                for i in range(2, degree + 1):
                    self.coeffs.data[c, i] = 0.15 * torch.randn(1) * (0.7 ** i)

        logit_eps = torch.log(torch.tensor(self.eps / (1.0 - self.eps)))
        self.register_buffer("alpha", -2.0 * logit_eps)
        self.register_buffer("beta", -logit_eps)

        if enable_spatial_variation:
            # Input channels:
            # 1  : base linear lambda
            # 4  : position encoding
            # 5  : content cues (|x0-y|, grad_x diff, grad_y diff, edge energy, anatomy mask)
            self.spatial_net = nn.Sequential(
                nn.Conv2d(10, 32, kernel_size=5, padding=2),
                nn.GroupNorm(8, 32),
                nn.GELU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.GroupNorm(8, 32),
                nn.GELU(),
                nn.Conv2d(32, channels, kernel_size=3, padding=1),
            )
            self.focus_head = nn.Sequential(
                nn.Conv2d(10, 16, kernel_size=3, padding=1),
                nn.GroupNorm(4, 16),
                nn.GELU(),
                nn.Conv2d(16, channels, kernel_size=3, padding=1),
            )
            self.register_buffer(
                "pos_encoding",
                self._create_position_encoding(*pos_hw),
                persistent=False,
            )
        else:
            self.spatial_net = None
            self.focus_head = None

    @staticmethod
    def _create_position_encoding(height: int, width: int) -> torch.Tensor:
        y_pos = torch.linspace(-1.0, 1.0, height).view(-1, 1).expand(-1, width)
        x_pos = torch.linspace(-1.0, 1.0, width).view(1, -1).expand(height, -1)
        return torch.stack(
            [
                torch.sin(torch.pi * y_pos),
                torch.cos(torch.pi * y_pos),
                torch.sin(torch.pi * x_pos),
                torch.cos(torch.pi * x_pos),
            ],
            dim=0,
        )

    def _calibrated_logistic(self, f: torch.Tensor) -> torch.Tensor:
        lam = torch.sigmoid(self.alpha * f - self.beta)
        return lam.clamp(self.eps, 1.0 - self.eps)

    @staticmethod
    def _normalize_per_sample(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        x_min = x.amin(dim=(-2, -1), keepdim=True)
        x_max = x.amax(dim=(-2, -1), keepdim=True)
        return (x - x_min) / (x_max - x_min + eps)

    @staticmethod
    def _finite_diff_h(x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        out[:, :, :, 1:] = x[:, :, :, 1:] - x[:, :, :, :-1]
        return out

    @staticmethod
    def _finite_diff_v(x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        out[:, :, 1:, :] = x[:, :, 1:, :] - x[:, :, :-1, :]
        return out

    def _blur(self, x: torch.Tensor) -> torch.Tensor:
        k = self.focus_blur_kernel
        if k <= 1:
            return x
        pad = k // 2
        return F.avg_pool2d(x, kernel_size=k, stride=1, padding=pad)

    def _anatomy_mask(self, x0: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        anatomy = torch.maximum(x0.abs(), y.abs()).mean(dim=1, keepdim=True)
        anatomy = self._normalize_per_sample(self._blur(anatomy))
        mask = torch.sigmoid(self.anatomy_slope * (anatomy - self.anatomy_threshold))
        return mask.clamp(0.0, 1.0)

    def _content_features(self, x0: torch.Tensor, y: torch.Tensor):
        diff = (x0 - y).abs().mean(dim=1, keepdim=True)
        dx = (self._finite_diff_h(x0) - self._finite_diff_h(y)).abs().mean(dim=1, keepdim=True)
        dy = (self._finite_diff_v(x0) - self._finite_diff_v(y)).abs().mean(dim=1, keepdim=True)
        edge = torch.sqrt(dx.pow(2) + dy.pow(2) + 1e-8)
        anatomy = self._anatomy_mask(x0, y)

        diff = self._normalize_per_sample(self._blur(diff))
        dx = self._normalize_per_sample(self._blur(dx))
        dy = self._normalize_per_sample(self._blur(dy))
        edge = self._normalize_per_sample(self._blur(edge))
        content_score = 0.45 * diff + 0.20 * dx + 0.20 * dy + 0.15 * edge
        content_score = self._normalize_per_sample(self._blur(content_score))
        gated_score = content_score * anatomy
        return torch.cat([diff, dx, dy, edge, anatomy], dim=1), gated_score, anatomy

    def _focus_map(self, x0: torch.Tensor, y: torch.Tensor, aux: torch.Tensor, gated_score: torch.Tensor, anatomy: torch.Tensor) -> torch.Tensor:
        if self.focus_head is None:
            focus = gated_score
        else:
            raw = self.focus_head(aux)
            focus_learned = torch.sigmoid(raw)
            focus_prior = gated_score.expand_as(focus_learned)
            anatomy_expand = anatomy.expand_as(focus_learned)
            focus = self.focus_mix * focus_learned + (1.0 - self.focus_mix) * focus_prior
            focus = focus * anatomy_expand
        focus = focus.pow(self.focus_power)
        return focus.clamp(0.0, 1.0)

    def forward(self, lam_base: torch.Tensor, x0: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Map scalar base schedule to a content-aware adaptive lambda field.

        Args:
            lam_base: [B] or [B,1,1,1], values in [0, 1]
            x0: current target-side reference [B, C, H, W]
            y: source image [B, C, H, W]
        Returns:
            lambda_hat: [B, C, H, W]
        """
        B, C, H, W = x0.shape
        if lam_base.dim() == 1:
            lam_base = lam_base.view(B, 1, 1, 1)
        elif lam_base.dim() != 4:
            lam_base = lam_base.view(B, 1, 1, 1)

        lam_plane = lam_base.expand(B, 1, H, W)

        if self.enable_spatial_variation and self.spatial_net is not None:
            pos = self.pos_encoding
            if pos.shape[-2:] != (H, W):
                pos = F.interpolate(
                    pos.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
                ).squeeze(0)
            pos = pos.unsqueeze(0).expand(B, -1, -1, -1)

            content, gated_score, anatomy = self._content_features(x0, y)
            aux = torch.cat([lam_plane, pos, content], dim=1)

            h = torch.sigmoid(self.spatial_net(aux))
            g = 2.0 * h - 1.0
            temporal_gain = (4.0 * lam_plane * (1.0 - lam_plane)).clamp(0.0, 1.0)
            f = lam_plane * (1.0 + temporal_gain * g * (1.0 - lam_plane))
            lam_adapt = self._calibrated_logistic(f)

            focus = self._focus_map(x0, y, aux, gated_score, anatomy)
            if focus.shape[1] == 1 and C > 1:
                focus = focus.expand(B, C, H, W)

            lam_linear = lam_base.expand(B, C, H, W)
            lam_hat = (1.0 - focus) * lam_linear + focus * lam_adapt
        else:
            lam_pow = [lam_plane ** i for i in range(self.coeffs.shape[1])]
            lam_hat = []
            for c in range(C):
                ch = sum(self.coeffs[c, i] * lam_pow[i] for i in range(self.coeffs.shape[1]))
                f0 = self.coeffs[c, 0]
                f1 = self.coeffs[c].sum()
                ch = (ch - f0) / (f1 - f0 + 1e-8)
                lam_hat.append(ch)
            lam_adapt = self._calibrated_logistic(torch.cat(lam_hat, dim=1))
            _, gated_score, anatomy = self._content_features(x0, y)
            focus = (gated_score * anatomy).pow(self.focus_power)
            if focus.shape[1] == 1 and C > 1:
                focus = focus.expand(B, C, H, W)
            lam_linear = lam_base.expand(B, C, H, W)
            lam_hat = (1.0 - focus) * lam_linear + focus * lam_adapt

        zero_mask = (lam_base <= 0).expand(B, C, H, W)
        one_mask = (lam_base >= 1).expand(B, C, H, W)
        lam_hat = torch.where(zero_mask, torch.zeros_like(lam_hat), lam_hat)
        lam_hat = torch.where(one_mask, torch.ones_like(lam_hat), lam_hat)
        return lam_hat.clamp(0.0, 1.0)


class DiffusionBridge(L.LightningModule):
    """Mathematically consistent adaptive bridge.

    The stochastic bridge remains exact: lambda_t only changes the mean path,
    while transition correlation and posterior are re-derived from the same chain.
    The only change here is that lambda_t is now content-aware and defaults back
    to the linear bridge on background / low-difference regions.
    """

    def __init__(
        self,
        n_steps,
        gamma,
        beta_start,
        beta_end,
        n_recursions,
        consistency_threshold,
        adaptive_bridge=True,
        lambda_degree=5,
        lambda_spatial=True,
        image_channels=1,
        lambda_eps=1e-4,
        lambda_anchor_weight=2e-3,
        lambda_tv_weight=1e-5,
        lambda_focus_weight=5e-4,
        lambda_focus_power=1.5,
        lambda_focus_blur=5,
        lambda_anatomy_threshold=0.10,
        lambda_anatomy_slope=10.0,
        lambda_focus_mix=0.75,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.gamma = gamma
        self.beta_start = beta_start
        self.beta_end = beta_end / n_steps
        self.n_recursions = n_recursions
        self.consistency_threshold = consistency_threshold
        self.adaptive_bridge = adaptive_bridge
        self.lambda_anchor_weight = lambda_anchor_weight
        self.lambda_tv_weight = lambda_tv_weight
        self.lambda_focus_weight = lambda_focus_weight

        self.betas = self._get_betas()

        s = np.cumsum(self.betas) ** 0.5
        s_bar = np.flip(np.cumsum(self.betas)) ** 0.5
        mu_x0, mu_y, _ = self.gaussian_product(s, s_bar)

        gamma = gamma * self.betas.sum()
        std = gamma * s / (s ** 2 + s_bar ** 2)

        self.register_buffer("s", torch.tensor(s, dtype=torch.float32))
        self.register_buffer("mu_x0", torch.tensor(mu_x0, dtype=torch.float32))
        self.register_buffer("mu_y", torch.tensor(mu_y, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

        rho = np.ones_like(std, dtype=np.float32)
        rho[1:] = mu_x0[1:] / np.clip(mu_x0[:-1], 1e-8, None)
        trans_var = np.zeros_like(std, dtype=np.float32)
        trans_var[1:] = std[1:] ** 2 - (rho[1:] ** 2) * (std[:-1] ** 2)
        if np.any(trans_var[1:] < -1e-8):
            raise ValueError("Invalid base schedule: negative transition variance.")
        trans_var = np.maximum(trans_var, 1e-8)

        self.register_buffer("rho", torch.tensor(rho, dtype=torch.float32))
        self.register_buffer("trans_var", torch.tensor(trans_var, dtype=torch.float32))

        if adaptive_bridge:
            self.lambda_net = AdaptiveBridgeLambda(
                channels=image_channels,
                degree=lambda_degree,
                enable_spatial_variation=lambda_spatial,
                eps=lambda_eps,
                focus_power=lambda_focus_power,
                focus_blur_kernel=lambda_focus_blur,
                anatomy_threshold=lambda_anatomy_threshold,
                anatomy_slope=lambda_anatomy_slope,
                focus_mix=lambda_focus_mix,
            )
        else:
            self.lambda_net = None

    def _shape(self, x: torch.Tensor):
        return [-1] + [1] * (x.ndim - 1)

    @staticmethod
    def _gradient_energy(x: torch.Tensor) -> torch.Tensor:
        dh = torch.zeros_like(x)
        dv = torch.zeros_like(x)
        dh[:, :, :, 1:] = x[:, :, :, 1:] - x[:, :, :, :-1]
        dv[:, :, 1:, :] = x[:, :, 1:, :] - x[:, :, :-1, :]
        return torch.sqrt(dh.pow(2) + dv.pow(2) + 1e-8)

    def _content_focus_target(self, x0: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = (x0 - y).abs().mean(dim=1, keepdim=True)
        edge = (self._gradient_energy(x0) - self._gradient_energy(y)).abs().mean(dim=1, keepdim=True)
        score = diff + 0.5 * edge
        score_min = score.amin(dim=(-2, -1), keepdim=True)
        score_max = score.amax(dim=(-2, -1), keepdim=True)
        return ((score - score_min) / (score_max - score_min + 1e-8)).clamp(0.0, 1.0)

    def _get_lambda(self, t: torch.Tensor, x0_like: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        lam_base = self.mu_y[t].view(self._shape(x0_like))
        if self.lambda_net is None:
            return lam_base.expand_as(x0_like)
        return self.lambda_net(lam_base, x0_like, y).to(dtype=x0_like.dtype)

    def _mean_from_lambda(self, lam_t: torch.Tensor, x0: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (1.0 - lam_t) * x0 + lam_t * y

    def q_sample(self, t, x0, y, detach=True):
        std = self.std[t].view(self._shape(x0))
        lam_t = self._get_lambda(t, x0, y)
        mean_t = self._mean_from_lambda(lam_t, x0, y)
        x_t = mean_t + std * torch.randn_like(x0)
        return x_t.detach() if detach else x_t

    def q_forward_step(self, t, x_tm1, x0, y):
        rho_t = self.rho[t].view(self._shape(x_tm1))
        trans_std_t = self.trans_var[t].view(self._shape(x_tm1)).sqrt()
        lam_t = self._get_lambda(t, x0, y)
        lam_tm1 = self._get_lambda(t - 1, x0, y)
        mean_t = self._mean_from_lambda(lam_t, x0, y)
        mean_tm1 = self._mean_from_lambda(lam_tm1, x0, y)
        cond_mean = rho_t * x_tm1 + (mean_t - rho_t * mean_tm1)
        return cond_mean + trans_std_t * torch.randn_like(x_tm1)

    def q_posterior(self, t, x_t, x0, y):
        shape = self._shape(x_t)
        sigma_t2 = (self.std[t].view(shape) ** 2).clamp(min=1e-8)
        sigma_tm12 = (self.std[t - 1].view(shape) ** 2).clamp(min=1e-8)
        rho_t = self.rho[t].view(shape)

        lam_t = self._get_lambda(t, x0, y)
        lam_tm1 = self._get_lambda(t - 1, x0, y)
        mean_t = self._mean_from_lambda(lam_t, x0, y)
        mean_tm1 = self._mean_from_lambda(lam_tm1, x0, y)

        k_t = rho_t * sigma_tm12 / sigma_t2
        post_var = (sigma_tm12 - (k_t ** 2) * sigma_t2).clamp(min=1e-8)
        post_mean = mean_tm1 + k_t * (x_t - mean_t)
        return post_mean + post_var.sqrt() * torch.randn_like(x_t)

    def lambda_regularization(self, t: torch.Tensor, x0: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.lambda_net is None or (
            self.lambda_anchor_weight <= 0 and self.lambda_tv_weight <= 0 and self.lambda_focus_weight <= 0
        ):
            return x0.new_zeros(())

        lam_base = self.mu_y[t].view(self._shape(x0)).expand_as(x0)
        lam_hat = self._get_lambda(t, x0, y)
        reg = x0.new_zeros(())

        if self.lambda_anchor_weight > 0:
            reg = reg + self.lambda_anchor_weight * F.mse_loss(lam_hat, lam_base)

        if self.lambda_tv_weight > 0:
            tv_h = (lam_hat[:, :, 1:, :] - lam_hat[:, :, :-1, :]).abs().mean()
            tv_w = (lam_hat[:, :, :, 1:] - lam_hat[:, :, :, :-1]).abs().mean()
            reg = reg + self.lambda_tv_weight * (tv_h + tv_w)

        if self.lambda_focus_weight > 0:
            focus_target = self._content_focus_target(x0, y)
            if focus_target.shape[1] == 1 and lam_hat.shape[1] > 1:
                focus_target = focus_target.expand_as(lam_hat)
            lam_delta = (lam_hat - lam_base).abs()
            reg = reg + self.lambda_focus_weight * F.l1_loss(lam_delta, focus_target * lam_delta.detach().amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6))

        return reg

    @torch.inference_mode()
    def sample_x0(self, y, generator):
        timesteps = torch.arange(self.n_steps, 0, -1, device=y.device)
        timesteps = timesteps.unsqueeze(1).repeat(1, y.shape[0])

        x_t = self.q_sample(timesteps[0], torch.zeros_like(y), y)

        for t in timesteps:
            x0_r = torch.zeros_like(x_t)
            for _ in range(self.n_recursions):
                x0_rp1 = generator(torch.cat((x_t, y), axis=1), t, x_r=x0_r)
                change = torch.abs(x0_rp1 - x0_r).mean(axis=0).max()
                if change < self.consistency_threshold:
                    break
                x0_r = x0_rp1

            x0_pred = x0_r
            x_t = self.q_posterior(t, x_t, x0_pred, y)

        return x0_pred

    def _get_betas(self):
        betas_len = self.n_steps + 1
        betas = np.linspace(self.beta_start ** 0.5, self.beta_end ** 0.5, betas_len) ** 2
        betas = np.append(0.0, betas).astype(np.float32)
        if betas_len % 2 == 1:
            betas = np.concatenate(
                [
                    betas[: betas_len // 2],
                    [betas[betas_len // 2]],
                    np.flip(betas[: betas_len // 2]),
                ]
            )
        else:
            betas = np.concatenate(
                [betas[: betas_len // 2], np.flip(betas[: betas_len // 2])]
            )
        return betas

    @staticmethod
    def gaussian_product(sigma1, sigma2):
        denom = sigma1 ** 2 + sigma2 ** 2
        mu1 = sigma2 ** 2 / denom
        mu2 = sigma1 ** 2 / denom
        var = (sigma1 ** 2 * sigma2 ** 2) / denom
        return mu1, mu2, var

    def vis_scheduler(self):
        plt.figure(figsize=(7, 3))
        plt.plot(self.std.cpu().numpy() ** 2, label=r"$\sigma_t^2$")
        plt.plot(self.mu_x0.cpu().numpy(), label=r"$\mu_{x_0}$")
        plt.plot(self.mu_y.cpu().numpy(), label=r"$\mu_y$")
        plt.plot(self.rho.cpu().numpy(), label=r"$\rho_t$")
        plt.legend()
        plt.tight_layout()
        plt.show()
