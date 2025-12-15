"""Abstract language-model interface the captioner talks to."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class LanguageModel(nn.Module, ABC):
    """A decoder-only LM that can consume pre-computed input embeddings.

    Accepting ``inputs_embeds`` is what makes audio conditioning possible: the
    captioner prepends audio prefix tokens to the text embeddings and feeds the
    whole sequence here.
    """

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """Embedding dimension the connector must project audio into."""

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Number of output tokens."""

    @abstractmethod
    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Look up token embeddings, shape ``[B, T, hidden_size]``."""

    @abstractmethod
    def forward(
        self,
        inputs_embeds: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return next-token logits of shape ``[B, T, vocab_size]``."""
