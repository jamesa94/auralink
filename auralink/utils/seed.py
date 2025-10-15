"""Reproducible seeding across random, numpy and torch."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed the standard RNGs used across the library.

    Args:
        seed: The integer seed.
        deterministic: If True, ask torch for deterministic kernels. This can
            slow things down and is off by default.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
