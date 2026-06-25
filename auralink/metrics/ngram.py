"""N-gram counting helpers used by BLEU and CIDEr."""

from __future__ import annotations

from collections import Counter
from typing import Counter as CounterType
from typing import List, Tuple


def get_ngrams(tokens: List[str], n: int) -> "CounterType[Tuple[str, ...]]":
    """Count the ``n``-grams in ``tokens``; empty when fewer than ``n`` tokens."""
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def count_ngrams_upto(tokens: List[str], max_n: int) -> dict:
    """Return a dict ``{n: Counter}`` for ``n`` in ``1..max_n``."""
    return {n: get_ngrams(tokens, n) for n in range(1, max_n + 1)}
