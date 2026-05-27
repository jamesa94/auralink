"""A torch ``Dataset`` yielding (waveform, tokenized caption) pairs."""

from __future__ import annotations

import zlib
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from auralink.audio.io import load_audio, synthetic_waveform
from auralink.data.manifest import ManifestItem
from auralink.llm.tokenizer import CharTokenizer

Loader = Callable[[ManifestItem, int], torch.Tensor]


def _stable_seed(key: str) -> int:
    return zlib.crc32(key.encode("utf-8")) % (2**31)


def default_loader(item: ManifestItem, sample_rate: int) -> torch.Tensor:
    """Load real audio if present, else a deterministic synthetic tone.

    Falling back to a synthetic waveform keeps examples and tests runnable
    without shipping audio files.
    """
    try:
        return load_audio(item.audio_path, sample_rate=sample_rate)
    except Exception:
        seed = _stable_seed(item.uid or item.audio_path)
        return synthetic_waveform(sample_rate, sample_rate=sample_rate, noise=0.01, seed=seed)


class CaptionDataset(Dataset):
    def __init__(
        self,
        items: List[ManifestItem],
        tokenizer: CharTokenizer,
        sample_rate: int = 16000,
        max_caption_len: int = 64,
        loader: Optional[Loader] = None,
        caption_index: int = 0,
    ) -> None:
        self.items = list(items)
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.max_caption_len = max_caption_len
        self.loader = loader or default_loader
        self.caption_index = caption_index

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        waveform = self.loader(item, self.sample_rate)
        caption = item.captions[self.caption_index % len(item.captions)] if item.captions else ""
        ids = self.tokenizer.encode(caption)[: self.max_caption_len]
        return {
            "waveform": waveform,
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "caption": caption,
        }
