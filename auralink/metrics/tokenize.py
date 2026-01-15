"""Caption normalisation and tokenisation shared by the text metrics."""

from __future__ import annotations

import re
from typing import List

_PUNCT = re.compile(r"[^\w\s]")
_WHITESPACE = re.compile(r"\s+")


def normalize_caption(text: str) -> str:
    """Lower-case, strip punctuation and collapse whitespace."""
    text = text.lower()
    text = _PUNCT.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def tokenize_caption(text: str) -> List[str]:
    """Split a caption into whitespace tokens after normalisation."""
    normalised = normalize_caption(text)
    return normalised.split() if normalised else []
