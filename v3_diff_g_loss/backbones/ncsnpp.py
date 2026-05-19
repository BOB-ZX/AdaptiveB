import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _num_groups(channels: int) -> int:
    for g in (32, 16, 8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class PixelNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t[None]
        t = t.float()
        half_dim = self.dim // 2
        emb_scale = math.log(10000.0) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb_scale)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class TimeMLP(nn.Module):
    def __init__(self, base_dim: int, out_dim: int):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(base_dim)
        self.net = nn.Sequential(
            nn.Linear(base_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(self.time_embed(t))


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, groups: int = 1):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, groups=groups),
            nn.GroupNorm(_num_groups(out_ch), out_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.conv(x)


class RecoveryStem(nn.Module):
    """Stable recovery stem. The recovery stream only sees x_t and x_r."""

    def __init__(self, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            ConvGNAct(2, out_ch, 3, 1),
            ConvGNAct(out_ch, out_ch, 3, 1),
        )

    def forward(self, x_t: torch.Tensor, x_r: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x_t, x_r], dim=1))


class YEncoder(nn.Module):
    def __init__(self, in_ch: int, feat_channels: List[int]):
        super().__init__()
        self.stages = nn.ModuleList()
        prev = in_ch
        for i, ch in enumerate(feat_channels):
            stride = 1 if i == 0 else 2
            self.stages.append(nn.Sequential(ConvGNAct(prev, ch, 3, stride), ConvGNAct(ch, ch, 3, 1)))
            prev = ch

    def forward(self, y: torch.Tensor) -> List[torch.Tensor]:
        feats = []
        h = y
        for stage in self.stages:
            h = stage(h)
            feats.append(h)
        return feats


class LambdaEncoder(nn.Module):
    """Multi-scale encoder for adaptive bridge strength lambda_hat."""

    def __init__(self, in_ch: int, feat_channels: List[int]):
        super().__init__()
        self.stages = nn.ModuleList()
        prev = in_ch
        for i, ch in enumerate(feat_channels):
            stride = 1 if i == 0 else 2
            self.stages.append(nn.Sequential(ConvGNAct(prev, ch, 3, stride), ConvGNAct(ch, ch, 3, 1)))
            prev = ch

    def forward(self, lambda_map: torch.Tensor) -> List[torch.Tensor]:
        feats = []
        h = lambda_map
        for stage in self.stages:
            h = stage(h)
            feats.append(h)
        return feats


class ZeroInitLambdaModulator(nn.Module):
    """Use lambda features to modulate source structural features.

    Zero initialization makes the model start from y-only structural guidance:
    guided_y = y_feat. During training lambda_hat learns to strengthen or weaken
    source guidance at each bridge position.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            ConvGNAct(channels, channels, 3, 1),
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, y_feat: torch.Tensor, lambda_feat: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        if lambda_feat.shape[-2:] != y_feat.shape[-2:]:
            lambda_feat = F.interpolate(lambda_feat, size=y_feat.shape[-2:], mode="bilinear", align_corners=False)
        gamma, beta = self.net(lambda_feat).chunk(2, dim=1)
        return y_feat * (1.0 + scale * gamma) + scale * beta


class StructureStrengthConditioner(nn.Module):
    """Package source structure and adaptive bridge strength into one condition."""

    def __init__(self, channels: List[int]):
        super().__init__()
        self.mods = nn.ModuleList([ZeroInitLambdaModulator(ch) for ch in channels])

    def forward(self, y_feats: List[torch.Tensor], lambda_feats: List[torch.Tensor], scale: float) -> List[torch.Tensor]:
        return [mod(yf, lf, scale=scale) for mod, yf, lf in zip(self.mods, y_feats, lambda_feats)]


class FiLMModulation(nn.Module):
    def __init__(self, feat_ch: int, cond_ch: int, time_dim: int, z_dim: int):
        super().__init__()
        hidden = max(feat_ch, cond_ch)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Linear(cond_ch + time_dim + z_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, feat_ch * 2),
        )

    def forward(self, x: torch.Tensor, cond_feat: torch.Tensor, t_emb: torch.Tensor, z_emb: torch.Tensor) -> torch.Tensor:
        if cond_feat.shape[-2:] != x.shape[-2:]:
            cond_feat = F.interpolate(cond_feat, size=x.shape[-2:], mode="bilinear", align_corners=False)
        pooled = self.pool(cond_feat).flatten(1)
        affine = self.net(torch.cat([pooled, t_emb, z_emb], dim=1))
        gamma, beta = affine.chunk(2, dim=1)
        return x * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]


class StructureAttention(nn.Module):
    def __init__(self, feat_ch: int, cond_ch: int):
        super().__init__()
        self.to_q = nn.Conv2d(feat_ch, feat_ch, 1)
        self.to_k = nn.Conv2d(cond_ch, feat_ch, 1)
        self.to_v = nn.Conv2d(cond_ch, feat_ch, 1)
        self.proj = nn.Conv2d(feat_ch, feat_ch, 1)

    def forward(self, x: torch.Tensor, cond_feat: torch.Tensor) -> torch.Tensor:
        if cond_feat.shape[-2:] != x.shape[-2:]:
            cond_feat = F.interpolate(cond_feat, size=x.shape[-2:], mode="bilinear", align_corners=False)
        q = self.to_q(x)
        k = self.to_k(cond_feat)
        v = self.to_v(cond_feat)
        attn = torch.sigmoid((q * k).sum(dim=1, keepdim=True) / math.sqrt(max(q.shape[1], 1)))
        return x + self.proj(attn * v)
    
class SelfAttention2d(nn.Module):
    """
    Bottleneck self-attention for U-Net features.

    This is different from StructureAttention:
    - StructureAttention: x attends to y structure feature
    - SelfAttention2d: h attends to itself spatially

    Zero-gamma initialized, so it is close to identity at the beginning.
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.GroupNorm(
            num_groups=min(32, channels // 4),
            num_channels=channels,
            eps=1e-6,
        )

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

        # zero-init output projection / residual scale
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, C, H, W = x.shape

        h = self.norm(x)
        qkv = self.qkv(h)

        q, k, v = torch.chunk(qkv, chunks=3, dim=1)

        q = q.view(B, self.num_heads, self.head_dim, H * W)
        k = k.view(B, self.num_heads, self.head_dim, H * W)
        v = v.view(B, self.num_heads, self.head_dim, H * W)

        q = q.permute(0, 1, 3, 2)  # [B, heads, HW, head_dim]
        k = k                       # [B, heads, head_dim, HW]
        v = v.permute(0, 1, 3, 2)  # [B, heads, HW, head_dim]

        attn = torch.matmul(q, k) * self.scale
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # [B, heads, HW, head_dim]
        out = out.permute(0, 1, 3, 2).contiguous()
        out = out.view(B, C, H, W)

        out = self.proj(out)

        return x + out

class GuidedResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_ch: int, time_dim: int, z_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(_num_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.z_proj = nn.Linear(z_dim, out_ch)
        nn.init.zeros_(self.z_proj.weight)
        nn.init.zeros_(self.z_proj.bias)
        self.norm2 = nn.GroupNorm(_num_groups(out_ch), out_ch)
        self.film = FiLMModulation(out_ch, cond_ch, time_dim, z_dim)
        self.struct_attn = StructureAttention(out_ch, cond_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor, cond_feat: torch.Tensor, t_emb: torch.Tensor, z_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = h + self.z_proj(z_emb)[:, :, None, None]
        h = F.silu(self.norm2(h))
        h = self.film(h, cond_feat, t_emb, z_emb)
        h = self.struct_attn(h, cond_feat)
        h = self.conv2(self.dropout(h))
        return (self.skip(x) + h) / math.sqrt(2.0)


class NCSNpp(nn.Module):
    """Lambda-guided structure-conditioned recursive generator.

    Main recovery stream: x_t + x_r only.
    External condition: y_encoder(y) provides structure; lambda_hat provides
    adaptive bridge strength and modulates the structural condition.
    """

    def __init__(
        self,
        self_recursion: bool = True,
        ch_mult: List[int] = [1, 1, 2, 2, 4, 4],
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        nf: int = 64,
        num_channels: int = 2,
        nz: int = 100,
        z_emb_dim: int = 256,
        n_mlp: int = 3,
        centered: bool = True,
        not_use_tanh: bool = False,
        lambda_channels: int = 1,
        lambda_guidance: bool = True,
        lambda_guidance_scale: float = 0.1,
        return_aux_outputs: bool = True,
        **unused_config,
    ):
        super().__init__()
        if num_channels < 2:
            raise ValueError("Expected x to contain [x_t, y] channels so y can be encoded as external structure.")
        self.self_recursion = self_recursion
        self.centered = centered
        self.not_use_tanh = not_use_tanh
        self.nf = nf
        self.nz = nz
        self.z_emb_dim = z_emb_dim
        self.num_res_blocks = num_res_blocks
        self.lambda_guidance = lambda_guidance
        self.lambda_guidance_scale = float(lambda_guidance_scale)
        self.return_aux_outputs = return_aux_outputs
        self.stage_channels = [nf * m for m in ch_mult]
        self.time_dim = nf * 4

        # XtXrFusion is intentionally removed. This is the only recovery input stem.
        self.input_stem = RecoveryStem(self.stage_channels[0])

        # External source structure and adaptive bridge-strength guidance.
        self.y_encoder = YEncoder(1, self.stage_channels)
        self.lambda_encoder = LambdaEncoder(lambda_channels, self.stage_channels)
        self.conditioner = StructureStrengthConditioner(self.stage_channels)
        self.time_mlp = TimeMLP(nf, self.time_dim)

        mapping_layers = [PixelNorm(), nn.Linear(nz, z_emb_dim), nn.SiLU()]
        for _ in range(n_mlp):
            mapping_layers += [nn.Linear(z_emb_dim, z_emb_dim), nn.SiLU()]
        self.z_transform = nn.Sequential(*mapping_layers)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        in_ch = self.stage_channels[0]
        for i, out_ch in enumerate(self.stage_channels):
            level_blocks = nn.ModuleList()
            for block_idx in range(num_res_blocks):
                block_in = in_ch if block_idx == 0 else out_ch
                level_blocks.append(GuidedResBlock(block_in, out_ch, out_ch, self.time_dim, z_emb_dim, dropout))
            self.down_blocks.append(level_blocks)
            in_ch = out_ch
            if i != len(self.stage_channels) - 1:
                self.downsamples.append(Downsample(in_ch))

        mid_ch = self.stage_channels[-1]
        self.mid_block1 = GuidedResBlock(mid_ch, mid_ch, mid_ch, self.time_dim, z_emb_dim, dropout)
        self.mid_attn = SelfAttention2d(mid_ch, num_heads=4)
        self.mid_block2 = GuidedResBlock(mid_ch, mid_ch, mid_ch, self.time_dim, z_emb_dim, dropout)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.aux_heads = nn.ModuleList()
        rev_channels = list(reversed(self.stage_channels))
        in_ch = rev_channels[0]
        for i, out_ch in enumerate(rev_channels):
            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                level_blocks.append(GuidedResBlock(in_ch + out_ch, out_ch, out_ch, self.time_dim, z_emb_dim, dropout))
                in_ch = out_ch
            self.up_blocks.append(level_blocks)
            self.aux_heads.append(nn.Conv2d(out_ch, 1, 3, padding=1))
            if i != len(rev_channels) - 1:
                self.upsamples.append(Upsample(in_ch))

        self.out_norm = nn.GroupNorm(_num_groups(in_ch), in_ch)
        self.out_conv = nn.Conv2d(in_ch, 1, 3, padding=1)

    def _split_inputs(self, x: torch.Tensor, x_r: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_t = x[:, :1]
        y = x[:, 1:2]
        if self.self_recursion:
            x_r = torch.zeros_like(x_t) if x_r is None else x_r[:, :1]
        else:
            x_r = torch.zeros_like(x_t)
        return x_t, y, x_r

    def _output_activation(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.not_use_tanh else torch.tanh(x)

    def forward(
        self,
        x: torch.Tensor,
        time_cond: torch.Tensor,
        x_r: Optional[torch.Tensor] = None,
        lambda_map: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        x_t, y, x_r = self._split_inputs(x, x_r)
        if not self.centered:
            x_t = 2.0 * x_t - 1.0
            y = 2.0 * y - 1.0
            x_r = 2.0 * x_r - 1.0

        if lambda_map is None:
            lambda_map = torch.zeros_like(y)
        else:
            lambda_map = lambda_map[:, :1].to(dtype=x_t.dtype)

        t_emb = self.time_mlp(time_cond)
        z = torch.randn(x_t.shape[0], self.nz, device=x_t.device, dtype=x_t.dtype)
        z_emb = self.z_transform(z)

        h = self.input_stem(x_t, x_r)
        y_feats = self.y_encoder(y)
        lambda_feats = self.lambda_encoder(lambda_map)
        cond_feats = self.conditioner(y_feats, lambda_feats, self.lambda_guidance_scale) if self.lambda_guidance else y_feats

        skips: List[torch.Tensor] = []
        for level, level_blocks in enumerate(self.down_blocks):
            cond = cond_feats[level]
            for block in level_blocks:
                h = block(h, cond, t_emb, z_emb)
                skips.append(h)
            if level < len(self.downsamples):
                h = self.downsamples[level](h)

        h = self.mid_block1(h, cond_feats[-1], t_emb, z_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, cond_feats[-1], t_emb, z_emb)

        aux_outputs: List[torch.Tensor] = []
        rev_cond_feats = list(reversed(cond_feats))
        upsample_idx = 0
        for level, level_blocks in enumerate(self.up_blocks):
            cond = rev_cond_feats[level]
            for block in level_blocks:
                skip = skips.pop()
                if skip.shape[-2:] != h.shape[-2:]:
                    h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
                h = torch.cat([h, skip], dim=1)
                h = block(h, cond, t_emb, z_emb)
            if return_aux and self.return_aux_outputs:
                aux_outputs.append(self._output_activation(self.aux_heads[level](h)))
            if level < len(self.upsamples):
                h = self.upsamples[upsample_idx](h)
                upsample_idx += 1

        out = self._output_activation(self.out_conv(F.silu(self.out_norm(h))))
        if return_aux:
            return {"final": out, "aux": aux_outputs}
        return out
