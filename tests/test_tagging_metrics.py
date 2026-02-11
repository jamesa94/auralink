import torch

from auralink.metrics.aggregate import evaluate_tags
from auralink.metrics.tagging import (
    average_precision,
    mean_average_precision,
    precision_recall_f1,
)


def test_perfect_ranking_average_precision():
    scores = torch.tensor([0.9, 0.8, 0.1, 0.05])
    targets = torch.tensor([1, 1, 0, 0])
    assert average_precision(scores, targets) > 0.99


def test_mean_average_precision_shape():
    torch.manual_seed(0)
    scores = torch.rand(10, 5)
    targets = (torch.rand(10, 5) > 0.5).float()
    mean, aps = mean_average_precision(scores, targets)
    assert len(aps) == 5
    assert 0.0 <= mean <= 1.0


def test_precision_recall_f1_perfect():
    scores = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
    targets = torch.tensor([[1, 0], [0, 1]])
    precision, recall, f1 = precision_recall_f1(scores, targets, threshold=0.5)
    assert precision == 1.0
    assert recall == 1.0
    assert f1 == 1.0


def test_evaluate_tags_keys():
    scores = torch.rand(6, 4)
    targets = (torch.rand(6, 4) > 0.5).float()
    out = evaluate_tags(scores, targets)
    assert set(out) == {"map", "precision", "recall", "f1"}
