"""Decoding configuration and the shared single-step helper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from auralink.llm.base import LanguageModel


@dataclass
class DecodeConfig:
    """Knobs for caption generation.

    ``strategy`` is one of ``"greedy"``, ``"beam"`` or ``"sample"``.
    """

    strategy: str = "greedy"
    max_new_tokens: int = 32
    beam_size: int = 4
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    length_penalty: float = 1.0


def next_token_logits(
    lm: LanguageModel,
    prefix_embeds: torch.Tensor,
    prefix_mask: Optional[torch.Tensor],
    generated_ids: torch.Tensor,
) -> torch.Tensor:
    """Run the LM over ``[audio prefix | generated tokens]`` and return last-step logits.

    Args:
        prefix_embeds: Audio prefix embeddings, ``[B, Q, D]``.
        prefix_mask: Validity mask for the prefix, ``[B, Q]`` or ``None``.
        generated_ids: Tokens decoded so far, ``[B, L]`` (at least the BOS token).

    Returns:
        Logits for the next token, ``[B, vocab_size]``.
    """
    token_embeds = lm.embed_tokens(generated_ids)
    embeds = torch.cat([prefix_embeds, token_embeds], dim=1)

    attention_mask: Optional[torch.Tensor] = None
    if prefix_mask is not None:
        token_mask = torch.ones(
            generated_ids.shape, dtype=torch.bool, device=generated_ids.device
        )
        attention_mask = torch.cat([prefix_mask, token_mask], dim=1)

    logits = lm(inputs_embeds=embeds, attention_mask=attention_mask)
    return logits[:, -1, :]
