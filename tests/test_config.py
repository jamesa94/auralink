from auralink.config.loader import dump_config, load_config, merge_overrides
from auralink.config.presets import get_preset, tiny_captioner_config
from auralink.model.config import CaptionerConfig


def test_load_from_dict_with_nested_override():
    cfg = load_config({"encoder": {"d_model": 64}, "max_caption_len": 10}, CaptionerConfig)
    assert cfg.encoder.d_model == 64
    assert cfg.max_caption_len == 10
    assert cfg.frontend.n_mels == 64  # default preserved


def test_yaml_roundtrip(tmp_path):
    cfg = tiny_captioner_config()
    path = tmp_path / "config.yaml"
    dump_config(cfg, str(path))
    loaded = load_config(str(path), CaptionerConfig)
    assert loaded.encoder.d_model == cfg.encoder.d_model
    assert loaded.connector.name == cfg.connector.name


def test_get_preset_with_kwargs():
    cfg = get_preset("tiny_tagger", num_classes=5)
    assert cfg.num_classes == 5


def test_merge_overrides_is_deep():
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    merged = merge_overrides(base, {"b": {"c": 9}})
    assert merged["b"] == {"c": 9, "d": 3}
    assert base["b"]["c"] == 2  # original untouched
