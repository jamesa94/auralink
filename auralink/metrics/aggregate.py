"""Convenience aggregators that bundle the individual metrics."""

from __future__ import annotations

from typing import Dict, Sequence

import torch

from auralink.metrics.bleu import corpus_bleu
from auralink.metrics.cider import compute_cider
from auralink.metrics.rouge import corpus_rouge_l
from auralink.metrics.tagging import mean_average_precision, precision_recall_f1


def evaluate_captions(
    candidates: Sequence[str],
    references_list: Sequence[Sequence[str]],
    max_n: int = 4,
) -> Dict[str, float]:
    """Compute BLEU-1..n, ROUGE-L and CIDEr for a set of predicted captions."""
    scores: Dict[str, float] = {}
    for k in range(1, max_n + 1):
        scores[f"bleu_{k}"] = corpus_bleu(candidates, references_list, n=k)
    scores["rouge_l"] = corpus_rouge_l(candidates, references_list)
    scores["cider"], _ = compute_cider(candidates, references_list)
    return scores


def evaluate_tags(
    scores: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> Dict[str, float]:
    """Compute mAP and micro P/R/F1 for multi-label tagging predictions."""
    map_score, _ = mean_average_precision(scores, targets)
    precision, recall, f1 = precision_recall_f1(scores, targets, threshold)
    return {"map": map_score, "precision": precision, "recall": recall, "f1": f1}
