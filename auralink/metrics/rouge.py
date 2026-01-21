"""ROUGE-L based on the longest common subsequence."""

from __future__ import annotations

from typing import List, Sequence

from auralink.metrics.tokenize import tokenize_caption


def _lcs_length(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[j - 1]))
        prev = curr
    return prev[-1]


def _rouge_l_pair(candidate: str, reference: str, beta: float) -> float:
    cand = tokenize_caption(candidate)
    ref = tokenize_caption(reference)
    if not cand or not ref:
        return 0.0
    lcs = _lcs_length(cand, ref)
    if lcs == 0:
        return 0.0
    precision = lcs / len(cand)
    recall = lcs / len(ref)
    beta_sq = beta * beta
    return ((1 + beta_sq) * precision * recall) / (recall + beta_sq * precision)


def rouge_l(candidate: str, references: Sequence[str], beta: float = 1.2) -> float:
    """ROUGE-L F-measure, taking the best-scoring reference."""
    return max((_rouge_l_pair(candidate, r, beta) for r in references), default=0.0)


def corpus_rouge_l(
    candidates: Sequence[str], references_list: Sequence[Sequence[str]], beta: float = 1.2
) -> float:
    """Mean ROUGE-L over a corpus."""
    if not candidates:
        return 0.0
    total = sum(rouge_l(c, refs, beta) for c, refs in zip(candidates, references_list))
    return total / len(candidates)
