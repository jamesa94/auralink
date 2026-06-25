import torch

from auralink.config.presets import tiny_tagger_config
from auralink.model.factory import build_tagger
from auralink.utils.checkpoint import load_checkpoint, load_state_into, save_checkpoint
from auralink.utils.seed import set_seed


def test_checkpoint_roundtrip(tmp_path):
    set_seed(0)
    model = build_tagger(tiny_tagger_config(8))
    path = tmp_path / "model.pt"
    save_checkpoint(str(path), model, config={"num_classes": 8})

    checkpoint = load_checkpoint(str(path))
    assert "state_dict" in checkpoint
    assert checkpoint["config"]["num_classes"] == 8

    other = build_tagger(tiny_tagger_config(8))
    load_state_into(other, str(path))
    for (_, a), (_, b) in zip(model.state_dict().items(), other.state_dict().items()):
        assert torch.equal(a, b)
