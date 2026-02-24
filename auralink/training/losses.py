"""Loss functions for the captioning and tagging objectives."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def caption_lm_loss(
    logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = 0
) -> torch.Tensor:
    """Cross-entropy over flattened ``[B, L, V]`` logits and ``[B, L]`` targets."""
    vocab = logits.shape[-1]
    return F.cross_entropy(
        logits.reshape(-1, vocab), targets.reshape(-1), ignore_index=ignore_index
    )


def tagging_bce_loss(
    logits: torch.Tensor, targets: torch.Tensor, pos_weight: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Binary cross-entropy for multi-label tagging."""
    return F.binary_cross_entropy_with_logits(logits, targets.float(), pos_weight=pos_weight)
