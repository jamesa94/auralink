"""A compact convolutional-stem + transformer audio encoder."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from auralink.encoders.base import AudioEncoder
from auralink.encoders.registry import register_encoder
from auralink.utils.masking import lengths_to_mask


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding added to the token sequence."""

    pe: torch.Tensor

    def __init__(self, dim: int, max_len: int = 4096) -> None:
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.shape[1]].unsqueeze(0).to(x.device)


@register_encoder("conv_transformer")
class ConvTransformerEncoder(AudioEncoder):
    """Downsample with a strided conv stem, then contextualise with a transformer.

    The conv stem reduces the frame rate by ``conv_stride ** n_conv`` which keeps
    the downstream connector and language model sequences short.
    """

    def __init__(
        self,
        input_dim: int = 64,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.1,
        conv_kernel: int = 3,
        conv_stride: int = 2,
        n_conv: int = 2,
    ) -> None:
        super().__init__()
        self._output_dim = d_model
        self.n_conv = n_conv
        self.conv_kernel = conv_kernel
        self.conv_stride = conv_stride
        self.conv_padding = conv_kernel // 2

        layers: List[nn.Module] = []
        in_channels = input_dim
        for _ in range(n_conv):
            layers.append(
                nn.Conv1d(
                    in_channels,
                    d_model,
                    kernel_size=conv_kernel,
                    stride=conv_stride,
                    padding=self.conv_padding,
                )
            )
            layers.append(nn.GELU())
            in_channels = d_model
        self.conv = nn.Sequential(*layers)

        self.pos_encoding = SinusoidalPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def downsample_factor(self) -> int:
        return self.conv_stride**self.n_conv

    def _downsampled_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        for _ in range(self.n_conv):
            lengths = (lengths + 2 * self.conv_padding - self.conv_kernel) // self.conv_stride + 1
        return lengths

    def forward(
        self, features: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = features.transpose(1, 2)  # [B, M, T]
        x = self.conv(x)  # [B, D, T']
        x = x.transpose(1, 2)  # [B, T', D]
        x = self.pos_encoding(x)
        out_len = x.shape[1]

        new_mask: Optional[torch.Tensor] = None
        key_padding_mask: Optional[torch.Tensor] = None
        if mask is not None:
            lengths = mask.sum(dim=1)
            new_lengths = self._downsampled_lengths(lengths).clamp(min=1, max=out_len)
            new_mask = lengths_to_mask(new_lengths, out_len).to(x.device)
            key_padding_mask = ~new_mask

        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x, new_mask
