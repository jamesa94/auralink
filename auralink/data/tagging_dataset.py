"""A torch ``Dataset`` yielding (waveform, multi-hot labels) pairs."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from auralink.data.caption_dataset import Loader, default_loader
from auralink.data.manifest import ManifestItem
from auralink.data.ontology import SoundEventOntology


class TaggingDataset(Dataset):
    def __init__(
        self,
        items: List[ManifestItem],
        ontology: SoundEventOntology,
        sample_rate: int = 16000,
        loader: Optional[Loader] = None,
    ) -> None:
        self.items = list(items)
        self.ontology = ontology
        self.sample_rate = sample_rate
        self.loader = loader or default_loader

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.items[idx]
        waveform = self.loader(item, self.sample_rate)
        labels = self.ontology.encode(item.labels)
        return {"waveform": waveform, "labels": labels}
