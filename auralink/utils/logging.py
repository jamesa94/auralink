"""A single, well-behaved logger factory for the package."""

from __future__ import annotations

import logging

_DEFAULT_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"


def get_logger(name: str = "auralink", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger that does not propagate to the root handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger
