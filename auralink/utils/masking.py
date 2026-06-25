"""Helpers for turning sequence lengths into boolean masks."""

from __future__ import annotations

import math
from typing import Optional

import torch


def lengths_to_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """Build a ``[B, max_len]`` boolean mask where ``True`` marks valid frames.

    Args:
        lengths: A 1-D integer tensor of valid lengths, shape ``[B]``.
        max_len: Output time dimension. Defaults to ``lengths.max()``.
    """
    lengths = torch.as_tensor(lengths)
    if max_len is None:
        max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0
    idx = torch.arange(max_len, device=lengths.device)
    return idx.unsqueeze(0) < lengths.unsqueeze(1)


def masked_mean(
    x: torch.Tensor, mask: Optional[torch.Tensor], dim: int = 1, eps: float = 1e-9
) -> torch.Tensor:
    """Average ``x`` over ``dim`` honouring a boolean ``mask`` (``True`` = valid)."""
    if mask is None:
        return x.mean(dim=dim)
    weights = mask.to(x.dtype).unsqueeze(-1)
    return (x * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(eps)


def conv_output_length(length: int, kernel: int, stride: int, padding: int) -> int:
    """Length of a 1-D convolution output (floor division, as in PyTorch)."""
    return (length + 2 * padding - kernel) // stride + 1


def downsample_lengths(lengths: torch.Tensor, factor: int) -> torch.Tensor:
    """Reduce lengths by an integer ``factor`` using ceiling division."""
    lengths = torch.as_tensor(lengths)
    return torch.ceil(lengths.float() / float(factor)).long()


def pad_to_multiple(x: torch.Tensor, multiple: int, dim: int = 1, value: float = 0.0) -> torch.Tensor:
    """Right-pad ``x`` along ``dim`` so that its size is a multiple of ``multiple``."""
    size = x.shape[dim]
    target = int(math.ceil(size / multiple) * multiple)
    if target == size:
        return x
    pad = [0, 0] * x.dim()
    # F.pad pads the last dim first; translate dim index into that layout.
    pad_index = (x.dim() - 1 - dim) * 2 + 1
    pad[pad_index] = target - size
    return torch.nn.functional.pad(x, pad, value=value)
