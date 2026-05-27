"""A deliberately small trainer that drives any model exposing ``training_step``."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from auralink.training.metrics_tracker import MetricTracker
from auralink.utils.logging import get_logger


class Trainer:
    """Run training/eval epochs for a captioner or tagger.

    The model only needs a ``training_step(batch) -> loss`` method, which both
    :class:`~auralink.model.captioner.AudioCaptioner` and
    :class:`~auralink.model.tagger.SoundEventTagger` provide.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler] = None,
        device: str = "cpu",
        grad_clip: Optional[float] = None,
        logger: Optional[Any] = None,
    ) -> None:
        # Typed as Any so static checkers accept the duck-typed training_step
        # provided by AudioCaptioner / SoundEventTagger.
        self.model: Any = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.grad_clip = grad_clip
        self.logger = logger or get_logger("auralink.trainer")
        self.global_step = 0

    def _move(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        tracker = MetricTracker()
        for batch in dataloader:
            batch = self._move(batch)
            loss = self.model.training_step(batch)
            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            tracker.update("loss", loss.item())
            self.global_step += 1
        return tracker.summary()

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        tracker = MetricTracker()
        for batch in dataloader:
            batch = self._move(batch)
            loss = self.model.training_step(batch)
            tracker.update("loss", loss.item())
        return tracker.summary()

    def fit(
        self,
        dataloader: DataLoader,
        epochs: int = 1,
        val_dataloader: Optional[DataLoader] = None,
    ) -> List[Dict[str, float]]:
        history: List[Dict[str, float]] = []
        for epoch in range(epochs):
            record: Dict[str, float] = {"epoch": float(epoch)}
            record.update({f"train_{k}": v for k, v in self.train_epoch(dataloader).items()})
            if val_dataloader is not None:
                record.update({f"val_{k}": v for k, v in self.evaluate(val_dataloader).items()})
            self.logger.info("epoch %d | %s", epoch, record)
            history.append(record)
        return history
