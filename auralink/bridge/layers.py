"""Reusable building blocks for connectors: RMSNorm, SwiGLU, cross-attention."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no mean subtraction), as used in LLaMA."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed * self.weight


class SwiGLU(nn.Module):
    """Gated feed-forward block: ``down(silu(gate(x)) * up(x))``."""

    def __init__(
        self, dim: int, hidden_dim: Optional[int] = None, out_dim: Optional[int] = None
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        out_dim = out_dim or dim
        self.w_gate = nn.Linear(dim, hidden_dim)
        self.w_up = nn.Linear(dim, hidden_dim)
        self.w_down = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class CrossAttention(nn.Module):
    """Thin wrapper over ``nn.MultiheadAttention`` for query->key/value attention."""

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out, _ = self.attn(
            query,
            key_value,
            key_value,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return out
