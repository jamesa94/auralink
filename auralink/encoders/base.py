"""Abstract base class shared by all audio encoders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn


class AudioEncoder(nn.Module, ABC):
    """Map log-mel features ``[B, T, M]`` to a contextual sequence ``[B, T', D]``.

    Implementations may downsample the time axis; when a frame mask is supplied
    they must return an updated mask aligned with the new sequence length.
    """

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Feature dimension ``D`` of the returned sequence."""

    @abstractmethod
    def forward(
        self, features: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return ``(sequence, mask)`` where ``mask`` marks valid frames or is ``None``."""
