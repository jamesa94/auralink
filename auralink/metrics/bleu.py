"""Corpus and sentence BLEU with closest-reference brevity penalty."""

from __future__ import annotations

import math
from collections import Counter
from typing import List, Sequence, Tuple

from auralink.metrics.ngram import get_ngrams
from auralink.metrics.tokenize import tokenize_caption


def _modified_precision(cand: List[str], refs: List[List[str]], n: int) -> Tuple[int, int]:
    cand_ngrams = get_ngrams(cand, n)
    if not cand_ngrams:
        return 0, 0
    max_ref: Counter = Counter()
    for ref in refs:
        for ngram, count in get_ngrams(ref, n).items():
            max_ref[ngram] = max(max_ref[ngram], count)
    clipped = sum(min(count, max_ref[ngram]) for ngram, count in cand_ngrams.items())
    return clipped, sum(cand_ngrams.values())


def _closest_ref_len(cand_len: int, ref_lens: Sequence[int]) -> int:
    return min(ref_lens, key=lambda r: (abs(r - cand_len), r))


def corpus_bleu(
    candidates: Sequence[str],
    references_list: Sequence[Sequence[str]],
    n: int = 4,
    smooth: bool = True,
) -> float:
    """BLEU-``n`` over a corpus of candidates each paired with multiple references."""
    clipped_totals = [0] * (n + 1)
    totals = [0] * (n + 1)
    cand_len_sum = 0
    ref_len_sum = 0

    for cand_str, refs_str in zip(candidates, references_list):
        cand = tokenize_caption(cand_str)
        refs = [tokenize_caption(r) for r in refs_str]
        cand_len_sum += len(cand)
        ref_len_sum += _closest_ref_len(len(cand), [len(r) for r in refs] or [0])
        for k in range(1, n + 1):
            clipped, total = _modified_precision(cand, refs, k)
            clipped_totals[k] += clipped
            totals[k] += total

    precisions: List[float] = []
    for k in range(1, n + 1):
        if totals[k] == 0:
            precisions.append(0.0)
        elif clipped_totals[k] == 0 and smooth:
            precisions.append(1.0 / (2 * totals[k]))
        else:
            precisions.append(clipped_totals[k] / totals[k])

    if any(p == 0.0 for p in precisions):
        geo_mean = 0.0
    else:
        geo_mean = math.exp(sum((1.0 / n) * math.log(p) for p in precisions))

    if cand_len_sum == 0:
        brevity = 0.0
    elif cand_len_sum > ref_len_sum:
        brevity = 1.0
    else:
        brevity = math.exp(1.0 - ref_len_sum / cand_len_sum)
    return brevity * geo_mean


def sentence_bleu(
    candidate: str, references: Sequence[str], n: int = 4, smooth: bool = True
) -> float:
    """BLEU-``n`` for a single candidate against its references."""
    return corpus_bleu([candidate], [references], n=n, smooth=smooth)
