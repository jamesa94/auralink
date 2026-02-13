"""Collate functions that pad variable-length audio and tokens into batches."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch


def _pad_waveforms(waveforms: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([w.shape[-1] for w in waveforms], dtype=torch.long)
    max_len = int(lengths.max()) if len(waveforms) else 0
    padded = torch.zeros(len(waveforms), max_len)
    for i, wav in enumerate(waveforms):
        padded[i, : wav.shape[-1]] = wav
    return padded, lengths


def collate_captions(batch: List[Dict[str, object]], pad_id: int = 0) -> Dict[str, object]:
    """Pad waveforms and caption token ids; build the text attention mask."""
    waveforms = [b["waveform"] for b in batch]
    wav, wav_lengths = _pad_waveforms(waveforms)

    id_lists = [b["input_ids"] for b in batch]
    max_len = max((len(ids) for ids in id_lists), default=0)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, ids in enumerate(id_lists):
        input_ids[i, : len(ids)] = ids
        attention_mask[i, : len(ids)] = True

    return {
        "waveform": wav,
        "wav_lengths": wav_lengths,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "captions": [b.get("caption", "") for b in batch],
    }


def collate_tags(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pad waveforms and stack multi-hot label vectors."""
    waveforms = [b["waveform"] for b in batch]
    wav, wav_lengths = _pad_waveforms(waveforms)
    labels = torch.stack([b["labels"] for b in batch])
    return {"waveform": wav, "wav_lengths": wav_lengths, "labels": labels}


class CaptionCollator:
    """Picklable collate callable for the captioner (``DataLoader(collate_fn=...)``)."""

    def __init__(self, pad_id: int = 0) -> None:
        self.pad_id = pad_id

    def __call__(self, batch: List[Dict[str, object]]) -> Dict[str, object]:
        return collate_captions(batch, self.pad_id)
