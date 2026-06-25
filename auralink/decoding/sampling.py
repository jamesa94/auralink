"""Temperature / top-k / nucleus (top-p) sampling."""

from __future__ import annotations

from typing import Optional

import torch

from auralink.decoding.base import next_token_logits
from auralink.llm.base import LanguageModel


def _filter_logits(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    if top_k and top_k > 0:
        k = min(top_k, logits.shape[-1])
        kth = torch.topk(logits, k, dim=-1).values[:, -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        to_remove = remove.scatter(1, sorted_idx, remove)
        logits = logits.masked_fill(to_remove, float("-inf"))
    return logits


@torch.no_grad()
def sample_decode(
    lm: LanguageModel,
    prefix_embeds: torch.Tensor,
    prefix_mask: Optional[torch.Tensor],
    bos_id: int,
    eos_id: int,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample token ids ``[B, L]`` (BOS stripped) from the LM distribution."""
    batch = prefix_embeds.shape[0]
    device = prefix_embeds.device
    generated = torch.full((batch, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(batch, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        logits = next_token_logits(lm, prefix_embeds, prefix_mask, generated)
        logits = logits / max(temperature, 1e-6)
        logits = _filter_logits(logits, top_k, top_p)
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(1)
        next_token = torch.where(finished, torch.full_like(next_token, eos_id), next_token)
        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
        finished = finished | (next_token == eos_id)
        if bool(finished.all()):
            break

    return generated[:, 1:]
