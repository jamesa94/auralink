"""Q-Former style connector: learned queries refined over several blocks."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from auralink.bridge.base import Connector
from auralink.bridge.layers import CrossAttention, RMSNorm, SwiGLU
from auralink.bridge.registry import register_connector


class QFormerBlock(nn.Module):
    """One block: query self-attention, audio cross-attention, then a SwiGLU FFN."""

    def __init__(self, dim: int, n_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = CrossAttention(dim, n_heads, dropout=dropout)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.norm3 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ff_dim, dim)

    def forward(
        self, queries: torch.Tensor, kv: torch.Tensor, key_padding_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        h = self.norm1(queries)
        self_out, _ = self.self_attn(h, h, h, need_weights=False)
        queries = queries + self_out

        h = self.norm2(queries)
        queries = queries + self.cross_attn(h, kv, key_padding_mask=key_padding_mask)

        h = self.norm3(queries)
        queries = queries + self.ffn(h)
        return queries


@register_connector("qformer")
class QFormerConnector(Connector):
    """A lightweight BLIP-2-style querying transformer.

    A fixed bank of learnable queries is refined across ``n_layers`` blocks that
    alternate self-attention and cross-attention over the audio sequence,
    yielding ``num_query_tokens`` information-dense prefix tokens.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_query_tokens: int = 16,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self._output_dim = output_dim
        self.num_query_tokens = num_query_tokens
        ff_dim = ff_dim or 4 * output_dim

        self.queries = nn.Parameter(torch.randn(num_query_tokens, output_dim) * 0.02)
        self.kv_proj = nn.Linear(input_dim, output_dim)
        self.blocks = nn.ModuleList(
            [QFormerBlock(output_dim, n_heads, ff_dim, dropout) for _ in range(n_layers)]
        )
        self.out_norm = RMSNorm(output_dim)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = x.shape[0]
        kv = self.kv_proj(x)
        queries = self.queries.unsqueeze(0).expand(batch, -1, -1).contiguous()
        key_padding_mask = ~mask if mask is not None else None

        for block in self.blocks:
            queries = block(queries, kv, key_padding_mask)
        queries = self.out_norm(queries)

        prefix_mask = torch.ones(batch, self.num_query_tokens, dtype=torch.bool, device=x.device)
        return queries, prefix_mask
