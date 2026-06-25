"""Abstract base class for audio -> language-model connectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn


class Connector(nn.Module, ABC):
    """Map an encoder sequence ``[B, T, D_in]`` to prefix tokens ``[B, Q, D_out]``.

    The returned prefix tokens are concatenated in front of the text token
    embeddings, acting as a soft prompt that conditions the language model on
    the audio. ``Q`` may be fixed (pooling / Q-Former) or proportional to the
    input length (temporal stacking).
    """

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimension ``D_out`` of the language-model embedding space."""

    @abstractmethod
    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(prefix_embeds [B, Q, D_out], prefix_mask [B, Q])``."""
