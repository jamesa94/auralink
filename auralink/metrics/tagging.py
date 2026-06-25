"""Multi-label tagging metrics: average precision, mAP and P/R/F1."""

from __future__ import annotations

from typing import List, Tuple

import torch


def average_precision(scores: torch.Tensor, targets: torch.Tensor) -> float:
    """Area under the precision-recall curve for a single class."""
    scores = torch.as_tensor(scores, dtype=torch.float)
    targets = torch.as_tensor(targets, dtype=torch.float)
    n_pos = targets.sum()
    if n_pos == 0:
        return 0.0
    order = torch.argsort(scores, descending=True)
    ordered = targets[order]
    cumulative_tp = torch.cumsum(ordered, dim=0)
    ranks = torch.arange(1, len(ordered) + 1, dtype=torch.float)
    precision_at_k = cumulative_tp / ranks
    return float((precision_at_k * ordered).sum() / n_pos)


def mean_average_precision(
    scores: torch.Tensor, targets: torch.Tensor
) -> Tuple[float, List[float]]:
    """Mean AP across classes for ``[N, C]`` score/target matrices."""
    scores = torch.as_tensor(scores, dtype=torch.float)
    targets = torch.as_tensor(targets, dtype=torch.float)
    num_classes = scores.shape[1]
    aps = [average_precision(scores[:, c], targets[:, c]) for c in range(num_classes)]
    return (sum(aps) / num_classes if num_classes else 0.0), aps


def precision_recall_f1(
    scores: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> Tuple[float, float, float]:
    """Micro-averaged precision, recall and F1 at a probability threshold."""
    preds = (torch.as_tensor(scores, dtype=torch.float) >= threshold).float()
    targets = torch.as_tensor(targets, dtype=torch.float)
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1
