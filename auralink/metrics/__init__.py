"""Pure-python evaluation metrics for captioning and sound-event tagging."""

from auralink.metrics.aggregate import evaluate_captions, evaluate_tags
from auralink.metrics.bleu import corpus_bleu, sentence_bleu
from auralink.metrics.cider import compute_cider
from auralink.metrics.rouge import corpus_rouge_l, rouge_l
from auralink.metrics.tagging import (
    average_precision,
    mean_average_precision,
    precision_recall_f1,
)
from auralink.metrics.tokenize import normalize_caption, tokenize_caption

__all__ = [
    "evaluate_captions",
    "evaluate_tags",
    "sentence_bleu",
    "corpus_bleu",
    "rouge_l",
    "corpus_rouge_l",
    "compute_cider",
    "average_precision",
    "mean_average_precision",
    "precision_recall_f1",
    "normalize_caption",
    "tokenize_caption",
]
