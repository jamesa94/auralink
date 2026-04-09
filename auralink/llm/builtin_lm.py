"""A tiny decoder-only transformer LM, built in so the stack runs offline.

It is not meant to produce good captions on its own; it exists so the whole
pipeline (encoder -> connector -> LM -> decoding -> metrics) can be exercised
and trained on a laptop without downloading any weights.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from auralink.llm.base import LanguageModel
from auralink.llm.registry import register_language_model


@register_language_model("tiny")
class TinyDecoderLM(LanguageModel):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: int = 256,
        max_len: int = 512,
        dropout: float = 0.1,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self._vocab_size = vocab_size
        self._hidden = d_model
        self.max_len = max_len
        self.pad_id = pad_id

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model,
            n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.layers = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # weight tying

    @property
    def hidden_size(self) -> int:
        return self._hidden

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.token_emb(input_ids)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # True marks positions that are NOT allowed to be attended to. A token
        # may attend to itself and everything before it, hence diagonal=1.
        ones = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        return torch.triu(ones, diagonal=1)

    def forward(
        self,
        inputs_embeds: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("provide either inputs_embeds or input_ids")
            inputs_embeds = self.embed_tokens(input_ids)

        seq_len = inputs_embeds.shape[1]
        positions = torch.arange(seq_len, device=inputs_embeds.device).clamp(max=self.max_len - 1)
        hidden = inputs_embeds + self.pos_emb(positions).unsqueeze(0)

        causal = self._causal_mask(seq_len, inputs_embeds.device)
        key_padding = None
        if attention_mask is not None:
            key_padding = ~attention_mask.bool()

        hidden = self.layers(hidden, mask=causal, src_key_padding_mask=key_padding)
        hidden = self.norm(hidden)
        return self.lm_head(hidden)
