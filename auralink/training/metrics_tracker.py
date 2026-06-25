"""Running-average meters for training loops."""

from __future__ import annotations

from typing import Dict


class AverageMeter:
    """Track a running mean of a scalar."""

    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def average(self) -> float:
        return self.sum / self.count if self.count else 0.0


class MetricTracker:
    """A small collection of named :class:`AverageMeter` instances."""

    def __init__(self) -> None:
        self.meters: Dict[str, AverageMeter] = {}

    def update(self, name: str, value: float, n: int = 1) -> None:
        self.meters.setdefault(name, AverageMeter()).update(value, n)

    def average(self, name: str) -> float:
        return self.meters[name].average if name in self.meters else 0.0

    def summary(self) -> Dict[str, float]:
        return {name: meter.average for name, meter in self.meters.items()}

    def reset(self) -> None:
        for meter in self.meters.values():
            meter.reset()
