"""SpecAugment-style time and frequency masking for log-mel features."""

from __future__ import annotations

import torch
import torch.nn as nn


class SpecAugment(nn.Module):
    """Apply random time and frequency masks to a ``[B, T, M]`` spectrogram.

    Masking is only applied in training mode, mirroring dropout semantics, so it
    is safe to leave in an evaluation pipeline.
    """

    def __init__(
        self,
        n_time_masks: int = 2,
        time_width: int = 10,
        n_freq_masks: int = 2,
        freq_width: int = 8,
        replace_with_zero: bool = True,
    ) -> None:
        super().__init__()
        self.n_time_masks = n_time_masks
        self.time_width = time_width
        self.n_freq_masks = n_freq_masks
        self.freq_width = freq_width
        self.replace_with_zero = replace_with_zero

    def _fill_value(self, spec: torch.Tensor) -> float:
        return 0.0 if self.replace_with_zero else float(spec.mean().item())

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return spec
        batch, n_frames, n_mels = spec.shape
        out = spec.clone()
        fill = self._fill_value(spec)
        for b in range(batch):
            for _ in range(self.n_time_masks):
                width = int(torch.randint(0, self.time_width + 1, (1,)).item())
                if width == 0 or width >= n_frames:
                    continue
                start = int(torch.randint(0, n_frames - width, (1,)).item())
                out[b, start : start + width, :] = fill
            for _ in range(self.n_freq_masks):
                width = int(torch.randint(0, self.freq_width + 1, (1,)).item())
                if width == 0 or width >= n_mels:
                    continue
                start = int(torch.randint(0, n_mels - width, (1,)).item())
                out[b, :, start : start + width] = fill
        return out
