"""A compact CIDEr-D implementation (tf-idf n-gram cosine with length penalty)."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

from auralink.metrics.ngram import get_ngrams
from auralink.metrics.tokenize import tokenize_caption

Vector = List[Dict[Tuple[str, ...], float]]


class CiderScorer:
    """Score captions with the CIDEr-D consensus metric.

    Document frequencies are estimated from the reference set, so scores are
    meaningful on a corpus rather than a single isolated caption.
    """

    def __init__(self, n: int = 4, sigma: float = 6.0) -> None:
        self.n = n
        self.sigma = sigma

    def _document_frequency(
        self, refs_tokens: List[List[List[str]]]
    ) -> Dict[Tuple[str, ...], float]:
        df: Dict[Tuple[str, ...], float] = defaultdict(float)
        for refs in refs_tokens:
            seen: set = set()
            for ref in refs:
                for k in range(1, self.n + 1):
                    seen.update(get_ngrams(ref, k))
            for ngram in seen:
                df[ngram] += 1.0
        return df

    def _vectorize(
        self, tokens: List[str], df: Dict[Tuple[str, ...], float], log_ref_count: float
    ) -> Tuple[Vector, List[float]]:
        vectors: Vector = [defaultdict(float) for _ in range(self.n)]
        norms = [0.0] * self.n
        for k in range(self.n):
            counts = get_ngrams(tokens, k + 1)
            total = max(sum(counts.values()), 1)
            for ngram, count in counts.items():
                idf = log_ref_count - math.log(max(df.get(ngram, 0.0), 1.0))
                vectors[k][ngram] = (count / total) * idf
            norms[k] = math.sqrt(sum(value * value for value in vectors[k].values()))
        return vectors, norms

    def _similarity(
        self,
        cand_vec: Vector,
        cand_norm: List[float],
        ref_vec: Vector,
        ref_norm: List[float],
        cand_len: int,
        ref_len: int,
    ) -> List[float]:
        per_n = [0.0] * self.n
        delta = cand_len - ref_len
        penalty = math.exp(-(delta * delta) / (2 * self.sigma * self.sigma))
        for k in range(self.n):
            denom = cand_norm[k] * ref_norm[k]
            if denom == 0.0:
                continue
            dot = sum(value * ref_vec[k].get(ngram, 0.0) for ngram, value in cand_vec[k].items())
            per_n[k] = (dot / denom) * penalty
        return per_n

    def compute(
        self, candidates: Sequence[str], references_list: Sequence[Sequence[str]]
    ) -> Tuple[float, List[float]]:
        cand_tokens = [tokenize_caption(c) for c in candidates]
        refs_tokens = [[tokenize_caption(r) for r in refs] for refs in references_list]
        df = self._document_frequency(refs_tokens)
        num_samples = len(candidates)
        # Use log(N + 1) rather than log(N): with a tiny corpus, n-grams that
        # appear in every sample would otherwise get a zero idf and collapse the
        # score to 0 even for an exact match.
        log_ref_count = math.log(num_samples + 1)

        scores: List[float] = []
        for cand, refs in zip(cand_tokens, refs_tokens):
            cand_vec, cand_norm = self._vectorize(cand, df, log_ref_count)
            per_ref = []
            for ref in refs:
                ref_vec, ref_norm = self._vectorize(ref, df, log_ref_count)
                sim = self._similarity(cand_vec, cand_norm, ref_vec, ref_norm, len(cand), len(ref))
                per_ref.append(sum(sim) / self.n)
            score = 10.0 * (sum(per_ref) / max(len(per_ref), 1))
            scores.append(score)

        mean = sum(scores) / max(num_samples, 1)
        return mean, scores


def compute_cider(
    candidates: Sequence[str],
    references_list: Sequence[Sequence[str]],
    n: int = 4,
    sigma: float = 6.0,
) -> Tuple[float, List[float]]:
    """Convenience wrapper returning ``(mean_cider, per_sample_scores)``."""
    return CiderScorer(n=n, sigma=sigma).compute(candidates, references_list)
