"""Batched beam search with length normalisation."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from auralink.decoding.base import next_token_logits
from auralink.llm.base import LanguageModel


@torch.no_grad()
def beam_search(
    lm: LanguageModel,
    prefix_embeds: torch.Tensor,
    prefix_mask: Optional[torch.Tensor],
    bos_id: int,
    eos_id: int,
    max_new_tokens: int = 32,
    beam_size: int = 4,
    length_penalty: float = 1.0,
) -> torch.Tensor:
    """Return the best beam per item, ``[B, L]`` (BOS stripped).

    The implementation expands the batch to ``B * beam_size`` hypotheses and
    re-scores with log-probabilities at every step. Finished beams keep their
    score and only ever emit EOS afterwards.
    """
    batch = prefix_embeds.shape[0]
    device = prefix_embeds.device
    vocab = lm.vocab_size
    q_len, dim = prefix_embeds.shape[1], prefix_embeds.shape[2]

    prefix = prefix_embeds.unsqueeze(1).expand(batch, beam_size, q_len, dim)
    prefix = prefix.reshape(batch * beam_size, q_len, dim)
    pmask: Optional[torch.Tensor] = None
    if prefix_mask is not None:
        pmask = prefix_mask.unsqueeze(1).expand(batch, beam_size, q_len)
        pmask = pmask.reshape(batch * beam_size, q_len)

    generated = torch.full((batch * beam_size, 1), bos_id, dtype=torch.long, device=device)
    scores = torch.full((batch, beam_size), float("-inf"), device=device)
    scores[:, 0] = 0.0
    scores = scores.view(-1)
    finished = torch.zeros(batch * beam_size, dtype=torch.bool, device=device)
    batch_offset = (torch.arange(batch, device=device) * beam_size).unsqueeze(1)

    for _ in range(max_new_tokens):
        logits = next_token_logits(lm, prefix, pmask, generated)
        log_probs = F.log_softmax(logits, dim=-1)
        # A finished beam contributes no new score and is forced to emit EOS.
        log_probs[finished] = float("-inf")
        log_probs[finished, eos_id] = 0.0

        candidate = scores.unsqueeze(1) + log_probs  # [B*beam, V]
        candidate = candidate.view(batch, beam_size * vocab)
        top_scores, top_idx = torch.topk(candidate, beam_size, dim=-1)
        beam_idx = torch.div(top_idx, vocab, rounding_mode="floor")
        token_idx = top_idx % vocab

        flat = (beam_idx + batch_offset).view(-1)
        generated = torch.cat([generated[flat], token_idx.view(-1, 1)], dim=1)
        scores = top_scores.view(-1)
        finished = finished[flat] | (token_idx.view(-1) == eos_id)
        if bool(finished.all()):
            break

    generated = generated.view(batch, beam_size, -1)
    scores = scores.view(batch, beam_size)
    lengths = (generated != eos_id).sum(dim=-1).clamp(min=1).float()
    normalised = scores / lengths.pow(length_penalty)
    best = normalised.argmax(dim=-1)
    chosen = generated[torch.arange(batch, device=device), best]
    return chosen[:, 1:]
