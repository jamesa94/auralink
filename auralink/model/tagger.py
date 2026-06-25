"""Multi-label sound-event tagger: frontend -> encoder -> pool -> classifier."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from auralink.audio.features import LogMelFrontend
from auralink.encoders.base import AudioEncoder
from auralink.encoders.pooling import AttentivePool, MeanPool
from auralink.utils.masking import lengths_to_mask


class SoundEventTagger(nn.Module):
    """Predict a multi-hot vector of sound events for a clip."""

    def __init__(
        self,
        frontend: LogMelFrontend,
        encoder: AudioEncoder,
        num_classes: int,
        pool: str = "attentive",
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.frontend = frontend
        self.encoder = encoder
        self.num_classes = num_classes

        dim = encoder.output_dim
        self.pool: nn.Module = AttentivePool(dim) if pool == "attentive" else MeanPool()
        hidden_dim = hidden_dim or dim
        self.classifier = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _feature_mask(
        self, feats: torch.Tensor, wav_lengths: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if wav_lengths is None:
            return None
        frame_lengths = self.frontend.num_frames(wav_lengths)
        return lengths_to_mask(frame_lengths, feats.shape[1]).to(feats.device)

    def encode(
        self, waveform: torch.Tensor, wav_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        feats = self.frontend(waveform)
        seq, mask = self.encoder(feats, self._feature_mask(feats, wav_lengths))
        return self.pool(seq, mask)

    def forward(
        self,
        waveform: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        wav_lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        pooled = self.encode(waveform, wav_lengths)
        logits = self.classifier(pooled)
        out: Dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            out["loss"] = F.binary_cross_entropy_with_logits(logits, labels.float())
        return out

    @torch.no_grad()
    def predict(
        self,
        waveform: torch.Tensor,
        threshold: float = 0.5,
        wav_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        was_training = self.training
        self.eval()
        logits = self.forward(waveform, wav_lengths=wav_lengths)["logits"]
        probs = torch.sigmoid(logits)
        self.train(was_training)
        return probs, probs >= threshold

    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.forward(
            batch["waveform"], labels=batch["labels"], wav_lengths=batch.get("wav_lengths")
        )["loss"]
