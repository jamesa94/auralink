"""A small, swappable sound-event label ontology."""

from __future__ import annotations

from typing import List, Sequence

import torch

# A compact default taxonomy; supply your own for AudioSet-scale work.
DEFAULT_EVENTS = [
    "speech",
    "dog",
    "cat",
    "music",
    "vehicle",
    "rain",
    "wind",
    "footsteps",
    "alarm",
    "applause",
    "engine",
    "bird",
]


class SoundEventOntology:
    """Map between event names and indices and build multi-hot label vectors."""

    def __init__(self, labels: Sequence[str]) -> None:
        self.labels: List[str] = list(labels)
        self.index = {label: i for i, label in enumerate(self.labels)}

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def num_classes(self) -> int:
        return len(self.labels)

    @classmethod
    def default(cls) -> "SoundEventOntology":
        return cls(DEFAULT_EVENTS)

    @classmethod
    def from_file(cls, path: str) -> "SoundEventOntology":
        with open(path, "r", encoding="utf-8") as handle:
            labels = [line.strip() for line in handle if line.strip()]
        return cls(labels)

    def encode(self, names: Sequence[str]) -> torch.Tensor:
        vector = torch.zeros(len(self.labels))
        for name in names:
            if name in self.index:
                vector[self.index[name]] = 1.0
        return vector

    def decode(self, multi_hot: Sequence[float], threshold: float = 0.5) -> List[str]:
        return [self.labels[i] for i, value in enumerate(multi_hot) if float(value) >= threshold]

    def label_name(self, idx: int) -> str:
        return self.labels[idx]
