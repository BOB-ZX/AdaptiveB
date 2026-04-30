from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from DWT_IDWT.DWT_IDWT_layer import DWT_2D, IDWT_2D


class PairWaveletBatchTransform(nn.Module):
    def __init__(
        self,
        wave: str = "haar",
        mode: str | None = None,
        J: int = 1,
        wavelet_scale: float = 2.0,
    ) -> None:
        super().__init__()
        if J != 1:
            raise ValueError(f"Only J=1 is supported for now, got {J}")
        if wave != "haar":
            raise ValueError(f"Only haar wave is supported for consistency with WaveDiff, got {wave}")
        if wavelet_scale <= 0:
            raise ValueError(f"wavelet_scale must be positive, got {wavelet_scale}")

        self.wave = wave
        self.mode = mode
        self.J = int(J)
        self.wavelet_scale = float(wavelet_scale)
        self.dwt = DWT_2D(wave)
        self.iwt = IDWT_2D(wave)

    def forward(self, batch: Tuple[Tensor, Tensor, Any] | Tuple[Tensor, Tensor]) -> Dict[str, Tensor]:
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            raise ValueError("batch must be a tuple or list like (x0, y, ...)")
        x0 = batch[0]
        y = batch[1]

        if x0.ndim != 4 or y.ndim != 4:
            raise ValueError(f"x0 and y must be BCHW tensors, got {tuple(x0.shape)} and {tuple(y.shape)}")
        if x0.shape != y.shape:
            raise ValueError(f"x0 and y must have the same shape, got {tuple(x0.shape)} and {tuple(y.shape)}")

        x0_ll, x0_lh, x0_hl, x0_hh = self.decompose(x0)
        y_ll, y_lh, y_hl, y_hh = self.decompose(y)

        x0_hf = self.pack_hf(x0_lh, x0_hl, x0_hh)
        y_hf = self.pack_hf(y_lh, y_hl, y_hh)
        x0_wavelet = self.pack_wavelet(x0_ll, x0_lh, x0_hl, x0_hh)
        y_wavelet = self.pack_wavelet(y_ll, y_lh, y_hl, y_hh)

        return {
            "x0": x0,
            "y": y,
            "x0_ll": x0_ll,
            "x0_lh": x0_lh,
            "x0_hl": x0_hl,
            "x0_hh": x0_hh,
            "x0_hf": x0_hf,
            "x0_hf_cat": x0_hf,
            "x0_wavelet": x0_wavelet,
            "x0_wavelet_cat": x0_wavelet,
            "y_ll": y_ll,
            "y_lh": y_lh,
            "y_hl": y_hl,
            "y_hh": y_hh,
            "y_hf": y_hf,
            "y_hf_cat": y_hf,
            "y_wavelet": y_wavelet,
            "y_wavelet_cat": y_wavelet,
        }

    def decompose(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        ll, lh, hl, hh = self.dwt(x)
        if self.wavelet_scale != 1.0:
            ll = ll / self.wavelet_scale
            lh = lh / self.wavelet_scale
            hl = hl / self.wavelet_scale
            hh = hh / self.wavelet_scale
        return ll, lh, hl, hh

    def reconstruct(self, ll: Tensor, lh: Tensor, hl: Tensor, hh: Tensor, output_size=None) -> Tensor:
        if self.wavelet_scale != 1.0:
            ll = ll * self.wavelet_scale
            lh = lh * self.wavelet_scale
            hl = hl * self.wavelet_scale
            hh = hh * self.wavelet_scale
        x = self.iwt(ll, lh, hl, hh)
        return self._match_output_size(x, output_size)

    @staticmethod
    def pack_hf(lh: Tensor, hl: Tensor, hh: Tensor) -> Tensor:
        return torch.cat([lh, hl, hh], dim=1)

    @staticmethod
    def unpack_hf(hf: Tensor, base_channels: int | None = None) -> Tuple[Tensor, Tensor, Tensor]:
        if base_channels is None:
            if hf.shape[1] % 3 != 0:
                raise ValueError(f"hf channels must be divisible by 3, got {hf.shape[1]}")
            base_channels = hf.shape[1] // 3
        if hf.shape[1] != 3 * base_channels:
            raise ValueError(f"Expected {3 * base_channels} HF channels, got {hf.shape[1]}")
        lh = hf[:, :base_channels]
        hl = hf[:, base_channels:2 * base_channels]
        hh = hf[:, 2 * base_channels:3 * base_channels]
        return lh, hl, hh

    @staticmethod
    def pack_wavelet(ll: Tensor, lh: Tensor, hl: Tensor, hh: Tensor) -> Tensor:
        return torch.cat([ll, lh, hl, hh], dim=1)

    @staticmethod
    def unpack_wavelet(wavelet: Tensor, base_channels: int | None = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if base_channels is None:
            if wavelet.shape[1] % 4 != 0:
                raise ValueError(f"wavelet channels must be divisible by 4, got {wavelet.shape[1]}")
            base_channels = wavelet.shape[1] // 4
        if wavelet.shape[1] != 4 * base_channels:
            raise ValueError(f"Expected {4 * base_channels} wavelet channels, got {wavelet.shape[1]}")
        ll = wavelet[:, :base_channels]
        lh = wavelet[:, base_channels:2 * base_channels]
        hl = wavelet[:, 2 * base_channels:3 * base_channels]
        hh = wavelet[:, 3 * base_channels:4 * base_channels]
        return ll, lh, hl, hh

    def inverse_from_parts(self, ll: Tensor, lh: Tensor, hl: Tensor, hh: Tensor, output_size=None) -> Tensor:
        return self.reconstruct(ll, lh, hl, hh, output_size=output_size)

    def inverse_from_hf_cat(self, ll: Tensor, hf: Tensor, base_channels: int | None = None, output_size=None) -> Tensor:
        lh, hl, hh = self.unpack_hf(hf, base_channels=base_channels)
        return self.reconstruct(ll, lh, hl, hh, output_size=output_size)

    def inverse_from_cat(self, wavelet: Tensor, base_channels: int | None = None, output_size=None) -> Tensor:
        ll, lh, hl, hh = self.unpack_wavelet(wavelet, base_channels=base_channels)
        return self.reconstruct(ll, lh, hl, hh, output_size=output_size)

    def inverse_from_wavelet_cat(self, wavelet: Tensor, base_channels: int | None = None, output_size=None) -> Tensor:
        return self.inverse_from_cat(wavelet, base_channels=base_channels, output_size=output_size)

    @staticmethod
    def _match_output_size(x: Tensor, output_size) -> Tensor:
        if output_size is None:
            return x
        if isinstance(output_size, torch.Tensor):
            output_size = tuple(int(v) for v in output_size.detach().cpu().flatten().tolist())
        if isinstance(output_size, torch.Size):
            output_size = tuple(output_size)
        if len(output_size) < 2:
            raise ValueError(f"output_size must contain height and width, got {output_size}")

        target_h = int(output_size[-2])
        target_w = int(output_size[-1])
        h, w = x.shape[-2:]

        x = x[..., :target_h, :target_w]
        pad_h = max(target_h - x.shape[-2], 0)
        pad_w = max(target_w - x.shape[-1], 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x
