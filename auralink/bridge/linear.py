"""Temporal-stacking connector: concatenate adjacent frames, then project."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from auralink.bridge.base import Connector
from auralink.bridge.layers import RMSNorm, SwiGLU
from auralink.bridge.registry import register_connector
from auralink.utils.masking import downsample_lengths, lengths_to_mask, pad_to_multiple


@register_connector("linear")
class LinearConnector(Connector):
    """Stack ``stack_size`` adjacent frames and map them to LM tokens.

    This is the cheapest connector and keeps a (downsampled) one-to-one
    correspondence between audio frames and prefix tokens, which works well for
    short clips. The number of prefix tokens is ``ceil(T / stack_size)``.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        stack_size: int = 2,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self._output_dim = output_dim
        self.stack_size = stack_size
        stacked_dim = input_dim * stack_size
        hidden_dim = hidden_dim or 4 * output_dim

        self.in_norm = RMSNorm(stacked_dim)
        self.proj_in = nn.Linear(stacked_dim, output_dim)
        self.ffn = SwiGLU(output_dim, hidden_dim, output_dim)
        self.out_norm = RMSNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, _, dim = x.shape
        x = pad_to_multiple(x, self.stack_size, dim=1)
        padded_len = x.shape[1]
        x = x.reshape(batch, padded_len // self.stack_size, dim * self.stack_size)

        x = self.in_norm(x)
        x = self.proj_in(x)
        x = x + self.dropout(self.ffn(x))
        x = self.out_norm(x)

        num_tokens = x.shape[1]
        if mask is not None:
            lengths = mask.sum(dim=1)
            new_lengths = downsample_lengths(lengths, self.stack_size).clamp(min=1, max=num_tokens)
            prefix_mask = lengths_to_mask(new_lengths, num_tokens).to(x.device)
        else:
            prefix_mask = torch.ones(batch, num_tokens, dtype=torch.bool, device=x.device)
        return x, prefix_mask
