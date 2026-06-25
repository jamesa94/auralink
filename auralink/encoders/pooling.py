"""Pooling layers that collapse a sequence ``[B, T, D]`` to a vector ``[B, D]``."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from auralink.utils.masking import masked_mean


class MeanPool(nn.Module):
    """Mask-aware average over the time axis."""

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return masked_mean(x, mask, dim=1)


class AttentivePool(nn.Module):
    """Learned attention pooling with a single query."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = self.score(x).squeeze(-1)  # [B, T]
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B, T, 1]
        return (x * weights).sum(dim=1)
