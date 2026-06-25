"""Loading configs from YAML/dicts and a few ready-made presets."""

from auralink.config.loader import config_to_dict, dump_config, load_config, merge_overrides
from auralink.config.presets import (
    base_captioner_config,
    get_preset,
    tiny_captioner_config,
    tiny_tagger_config,
)

__all__ = [
    "load_config",
    "dump_config",
    "config_to_dict",
    "merge_overrides",
    "tiny_captioner_config",
    "base_captioner_config",
    "tiny_tagger_config",
    "get_preset",
]
