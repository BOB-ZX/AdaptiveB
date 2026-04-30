import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict

# -----------------------------------------------------------------------------
# Drop-in replacement for SelfRDB/backbones/ncsnpp.py
#
# Key idea:
#   - main.py in SelfRDB still calls:
#         generator(torch.cat((x_t, y), axis=1), t, x_r=x0_r)
#   - this replacement keeps the SAME constructor name and SAME forward signature:
#         NCSNpp(...).forward(x, time_cond, x_r=None)
#   - internally it splits x into:
#         x_t = x[:, :1]
#         y   = x[:, 1:2]
#   - x_r is used only for collaborative denoising with x_t
#   - y never enters the denoising branch as a generation input; it only modulates
#     the denoising process with FiLM + structure attention.
# -----------------------------------------------------------------------------


def _num_groups(channels: int) -> int:
    for g in (32, 16, 8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t[None]
        t = t.float()
        half_dim = self.dim // 2
        device = t.device
        emb_scale = math.log(10000.0) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
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
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=pad),
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
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        return self.conv(x)


class XtXrFusion(nn.Module):
    """Collaborative denoising stem for x_t and x_r only."""
    def __init__(self, out_ch: int):
        super().__init__()
        self.stem = nn.Sequential(
            ConvGNAct(2, out_ch, 3, 1),
            ConvGNAct(out_ch, out_ch, 3, 1),
        )
        self.diff_gate = nn.Sequential(
            nn.Conv2d(1, out_ch, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x_t: torch.Tensor, x_r: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(x_t - x_r)
        h = self.stem(torch.cat([x_t, x_r], dim=1))
        gate = self.diff_gate(diff)
        return h * (1.0 + gate)


class FiLMModulation(nn.Module):
    """Generate gamma/beta from y-features and timestep embedding."""
    def __init__(self, feat_ch: int, cond_ch: int, time_dim: int):
        super().__init__()
        hidden = max(feat_ch, cond_ch)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Linear(cond_ch + time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, feat_ch * 2),
        )

    def forward(self, x: torch.Tensor, y_feat: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(y_feat).flatten(1)
        affine = self.net(torch.cat([pooled, t_emb], dim=1))
        gamma, beta = affine.chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return x * (1.0 + gamma) + beta


class StructureAttention(nn.Module):
    """Linear-cost structure attention from y to x.

    Uses per-pixel query-key similarity instead of full HW x HW attention,
    which is much cheaper and more stable on 256x256 medical slices.
    """
    def __init__(self, feat_ch: int, cond_ch: int):
        super().__init__()
        self.to_q = nn.Conv2d(feat_ch, feat_ch, 1)
        self.to_k = nn.Conv2d(cond_ch, feat_ch, 1)
        self.to_v = nn.Conv2d(cond_ch, feat_ch, 1)
        self.proj = nn.Conv2d(feat_ch, feat_ch, 1)

    def forward(self, x: torch.Tensor, y_feat: torch.Tensor) -> torch.Tensor:
        if y_feat.shape[-2:] != x.shape[-2:]:
            y_feat = F.interpolate(y_feat, size=x.shape[-2:], mode='bilinear', align_corners=False)
        q = self.to_q(x)
        k = self.to_k(y_feat)
        v = self.to_v(y_feat)
        attn = torch.sigmoid((q * k).sum(dim=1, keepdim=True) / math.sqrt(max(q.shape[1], 1)))
        out = self.proj(attn * v)
        return x + out


class GuidedResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_ch: int, time_dim: int, dropout: float = 0.0):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.norm1 = nn.GroupNorm(_num_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)

        self.norm2 = nn.GroupNorm(_num_groups(out_ch), out_ch)
        self.film = FiLMModulation(out_ch, cond_ch, time_dim)
        self.struct_attn = StructureAttention(out_ch, cond_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor, y_feat: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = F.silu(self.norm2(h))
        h = self.film(h, y_feat, t_emb)
        h = self.struct_attn(h, y_feat)
        h = self.conv2(self.dropout(h))
        return self.skip(x) + h


class YEncoder(nn.Module):
    def __init__(self, in_ch: int, feat_channels: List[int]):
        super().__init__()
        self.stages = nn.ModuleList()
        prev = in_ch
        for i, ch in enumerate(feat_channels):
            stride = 1 if i == 0 else 2
            self.stages.append(
                nn.Sequential(
                    ConvGNAct(prev, ch, 3, stride),
                    ConvGNAct(ch, ch, 3, 1),
                )
            )
            prev = ch

    def forward(self, y: torch.Tensor) -> List[torch.Tensor]:
        feats = []
        h = y
        for stage in self.stages:
            h = stage(h)
            feats.append(h)
        return feats


class NCSNpp(nn.Module):
    """Drop-in SelfRDB generator replacement.

    Public interface intentionally kept compatible with the original NCSNpp:
      - constructor name: NCSNpp
      - forward signature: forward(x, time_cond, x_r=None)

    Expected input from existing SelfRDB main.py:
      x = cat([x_t, y], dim=1)  # 2 channels
      x_r = recursive estimate     # 1 channel or None
    """

    def __init__(
        self,
        self_recursion: bool = True,
        z_emb_dim: int = 256,  # kept for config compatibility
        ch_mult: List[int] = [1, 1, 2, 2, 4, 4],
        num_res_blocks: int = 2,
        attn_resolutions: List[int] = [16],  # kept for config compatibility
        dropout: float = 0.0,
        resamp_with_conv: bool = True,  # kept for config compatibility
        image_size: int = 256,
        conditional: bool = True,  # kept for config compatibility
        fir: bool = True,  # kept for config compatibility
        fir_kernel: List[int] = [1, 3, 3, 1],  # kept for config compatibility
        skip_rescale: bool = True,  # kept for config compatibility
        resblock_type: str = 'biggan',  # kept for config compatibility
        progressive: str = 'none',  # kept for config compatibility
        progressive_input: str = 'residual',  # kept for config compatibility
        embedding_type: str = 'positional',  # kept for config compatibility
        combine_method: str = 'sum',  # kept for config compatibility
        fourier_scale: int = 16,  # kept for config compatibility
        nf: int = 64,
        num_channels: int = 2,
        nz: int = 100,  # kept for config compatibility
        n_mlp: int = 3,  # kept for config compatibility
        centered: bool = True,
        not_use_tanh: bool = False,
    ):
        super().__init__()
        del z_emb_dim, attn_resolutions, resamp_with_conv, conditional
        del fir, fir_kernel, skip_rescale, resblock_type, progressive
        del progressive_input, embedding_type, combine_method, fourier_scale
        del nz, n_mlp, image_size  # all kept only for checkpoint/config compatibility

        if num_channels < 2:
            raise ValueError(
                'This replacement expects generator input with at least 2 channels: [x_t, y]. '
                f'Got num_channels={num_channels}.'
            )

        self.self_recursion = self_recursion
        self.centered = centered
        self.not_use_tanh = not_use_tanh
        self.nf = nf
        self.num_res_blocks = num_res_blocks
        self.stage_channels = [nf * m for m in ch_mult]
        self.time_dim = nf * 4

        # x_t + x_r collaborative denoising branch
        self.xtxr_fusion = XtXrFusion(self.stage_channels[0])

        # y-only structural guidance branch
        self.y_encoder = YEncoder(1, self.stage_channels)

        self.time_mlp = TimeMLP(nf, self.time_dim)

        # Down path
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        in_ch = self.stage_channels[0]
        for i, out_ch in enumerate(self.stage_channels):
            level_blocks = nn.ModuleList()
            for block_idx in range(num_res_blocks):
                block_in = in_ch if block_idx == 0 else out_ch
                level_blocks.append(GuidedResBlock(block_in, out_ch, out_ch, self.time_dim, dropout))
            self.down_blocks.append(level_blocks)
            in_ch = out_ch
            if i != len(self.stage_channels) - 1:
                self.downsamples.append(Downsample(in_ch))

        # Mid blocks
        mid_ch = self.stage_channels[-1]
        self.mid_block1 = GuidedResBlock(mid_ch, mid_ch, mid_ch, self.time_dim, dropout)
        self.mid_block2 = GuidedResBlock(mid_ch, mid_ch, mid_ch, self.time_dim, dropout)

        # Up path
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        rev_channels = list(reversed(self.stage_channels))
        in_ch = rev_channels[0]
        for i, out_ch in enumerate(rev_channels):
            level_blocks = nn.ModuleList()
            n_blocks = num_res_blocks
            for _ in range(n_blocks):
                level_blocks.append(GuidedResBlock(in_ch + out_ch, out_ch, out_ch, self.time_dim, dropout))
                in_ch = out_ch
            self.up_blocks.append(level_blocks)
            if i != len(rev_channels) - 1:
                self.upsamples.append(Upsample(in_ch))

        self.out_norm = nn.GroupNorm(_num_groups(in_ch), in_ch)
        self.out_conv = nn.Conv2d(in_ch, 1, 3, padding=1)

    def _split_inputs(self, x: torch.Tensor, x_r: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_t = x[:, :1]
        y = x[:, 1:2]

        if self.self_recursion:
            if x_r is None:
                x_r = torch.zeros_like(x_t)
            else:
                x_r = x_r[:, :1]
        else:
            x_r = torch.zeros_like(x_t)
        return x_t, y, x_r

    def forward(self, x: torch.Tensor, time_cond: torch.Tensor, x_r: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_t, y, x_r = self._split_inputs(x, x_r)
        t_emb = self.time_mlp(time_cond)

        # Collaborative denoising input: only x_t and x_r
        h = self.xtxr_fusion(x_t, x_r)

        # Structural guidance comes only from y
        y_feats = self.y_encoder(y)

        # Down path
        skips: List[torch.Tensor] = []
        for level, level_blocks in enumerate(self.down_blocks):
            y_feat = y_feats[level]
            for block in level_blocks:
                h = block(h, y_feat, t_emb)
                skips.append(h)
            if level < len(self.downsamples):
                h = self.downsamples[level](h)

        # Mid
        h = self.mid_block1(h, y_feats[-1], t_emb)
        h = self.mid_block2(h, y_feats[-1], t_emb)

        # Up path
        rev_y_feats = list(reversed(y_feats))
        upsample_idx = 0
        for level, level_blocks in enumerate(self.up_blocks):
            y_feat = rev_y_feats[level]
            for block in level_blocks:
                skip = skips.pop()
                if skip.shape[-2:] != h.shape[-2:]:
                    h = F.interpolate(h, size=skip.shape[-2:], mode='nearest')
                h = torch.cat([h, skip], dim=1)
                h = block(h, y_feat, t_emb)
            if level < len(self.upsamples):
                h = self.upsamples[upsample_idx](h)
                upsample_idx += 1

        out = self.out_conv(F.silu(self.out_norm(h)))
        if not self.not_use_tanh:
            out = torch.tanh(out)
        return out
