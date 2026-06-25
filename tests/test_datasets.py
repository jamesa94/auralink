import torch

from auralink.data.caption_dataset import CaptionDataset
from auralink.data.manifest import ManifestItem
from auralink.data.ontology import SoundEventOntology
from auralink.data.tagging_dataset import TaggingDataset
from auralink.llm.tokenizer import CharTokenizer


def _items():
    return [
        ManifestItem(audio_path="missing0.wav", captions=["a dog barks"], labels=["dog"], uid="0"),
        ManifestItem(
            audio_path="missing1.wav", captions=["rain falling"], labels=["rain"], uid="1"
        ),
    ]


def test_caption_dataset_item():
    tok = CharTokenizer.default()
    ds = CaptionDataset(_items(), tok, sample_rate=4000)
    sample = ds[0]
    assert sample["waveform"].dim() == 1
    assert sample["input_ids"].dim() == 1
    assert sample["caption"] == "a dog barks"


def test_tagging_dataset_encodes_labels():
    onto = SoundEventOntology.default()
    ds = TaggingDataset(_items(), onto, sample_rate=4000)
    sample = ds[0]
    assert sample["labels"].shape == (onto.num_classes,)
    assert sample["labels"][onto.index["dog"]] == 1.0


def test_synthetic_fallback_is_deterministic():
    tok = CharTokenizer.default()
    ds = CaptionDataset(_items(), tok, sample_rate=4000)
    assert torch.equal(ds[0]["waveform"], ds[0]["waveform"])
    assert not torch.equal(ds[0]["waveform"], ds[1]["waveform"])
