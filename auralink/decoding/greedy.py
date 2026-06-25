"""Greedy (argmax) decoding."""

from __future__ import annotations

from typing import Optional

import torch

from auralink.decoding.base import next_token_logits
from auralink.llm.base import LanguageModel


@torch.no_grad()
def greedy_decode(
    lm: LanguageModel,
    prefix_embeds: torch.Tensor,
    prefix_mask: Optional[torch.Tensor],
    bos_id: int,
    eos_id: int,
    max_new_tokens: int = 32,
) -> torch.Tensor:
    """Decode greedily, returning generated token ids ``[B, L]`` (BOS stripped)."""
    batch = prefix_embeds.shape[0]
    device = prefix_embeds.device
    generated = torch.full((batch, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(batch, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        logits = next_token_logits(lm, prefix_embeds, prefix_mask, generated)
        next_token = logits.argmax(dim=-1)
        next_token = torch.where(finished, torch.full_like(next_token, eos_id), next_token)
        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
        finished = finished | (next_token == eos_id)
        if bool(finished.all()):
            break

    return generated[:, 1:]
