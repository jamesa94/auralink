import torch

from auralink.audio.io import synthetic_waveform
from auralink.data.collate import CaptionCollator, collate_captions, collate_tags


def test_collate_captions_pads_audio_and_tokens():
    batch = [
        {
            "waveform": synthetic_waveform(3000, seed=0),
            "input_ids": torch.tensor([1, 2, 3]),
            "caption": "x",
        },
        {
            "waveform": synthetic_waveform(5000, seed=1),
            "input_ids": torch.tensor([1, 2, 3, 4, 5]),
            "caption": "y",
        },
    ]
    out = collate_captions(batch, pad_id=0)
    assert out["waveform"].shape == (2, 5000)
    assert out["input_ids"].shape == (2, 5)
    assert int(out["attention_mask"][0].sum()) == 3
    assert out["wav_lengths"].tolist() == [3000, 5000]
    assert out["captions"] == ["x", "y"]


def test_caption_collator_callable():
    collate = CaptionCollator(pad_id=0)
    batch = [
        {
            "waveform": synthetic_waveform(1000, seed=0),
            "input_ids": torch.tensor([1, 2]),
            "caption": "z",
        }
    ]
    assert collate(batch)["input_ids"].shape == (1, 2)


def test_collate_tags_stacks_labels():
    batch = [
        {"waveform": synthetic_waveform(2000, seed=0), "labels": torch.zeros(8)},
        {"waveform": synthetic_waveform(2500, seed=1), "labels": torch.ones(8)},
    ]
    out = collate_tags(batch)
    assert out["waveform"].shape == (2, 2500)
    assert out["labels"].shape == (2, 8)
