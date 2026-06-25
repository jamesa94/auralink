"""Pooling connector: a fixed bank of learned queries attends over the audio."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from auralink.bridge.base import Connector
from auralink.bridge.layers import CrossAttention, RMSNorm, SwiGLU
from auralink.bridge.registry import register_connector


@register_connector("pooling")
class PoolingConnector(Connector):
    """Compress an audio sequence into a fixed number of prefix tokens.

    A small set of learned query vectors cross-attend over the encoder output,
    producing ``num_query_tokens`` prefix tokens regardless of clip length.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_query_tokens: int = 8,
        n_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self._output_dim = output_dim
        self.num_query_tokens = num_query_tokens
        self.queries = nn.Parameter(torch.randn(num_query_tokens, output_dim) * 0.02)
        self.kv_proj = nn.Linear(input_dim, output_dim)
        self.attn = CrossAttention(output_dim, n_heads, dropout=dropout)
        self.norm = RMSNorm(output_dim)
        self.ffn = SwiGLU(output_dim, out_dim=output_dim)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = x.shape[0]
        kv = self.kv_proj(x)
        queries = self.queries.unsqueeze(0).expand(batch, -1, -1)
        key_padding_mask = ~mask if mask is not None else None

        attended = self.attn(queries, kv, key_padding_mask=key_padding_mask)
        out = self.norm(queries + attended)
        out = out + self.ffn(out)

        prefix_mask = torch.ones(batch, self.num_query_tokens, dtype=torch.bool, device=x.device)
        return out, prefix_mask
