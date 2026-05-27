"""Load dataclass configs from YAML files or plain dicts."""

from __future__ import annotations

import dataclasses
import typing
from typing import Any, Dict, Type, TypeVar, Union

import yaml

T = TypeVar("T")


def _from_dict(cls: Any, data: Dict[str, Any]) -> Any:
    if not dataclasses.is_dataclass(cls):
        return data
    hints = typing.get_type_hints(cls)
    kwargs: Dict[str, Any] = {}
    for field in dataclasses.fields(cls):
        if field.name not in data:
            continue
        field_type: Any = hints.get(field.name)
        value = data[field.name]
        if dataclasses.is_dataclass(field_type) and isinstance(value, dict):
            kwargs[field.name] = _from_dict(field_type, value)
        else:
            kwargs[field.name] = value
    constructor: Any = cls
    return constructor(**kwargs)


def load_config(source: Union[str, Dict[str, Any]], config_cls: Type[T]) -> T:
    """Build a (possibly nested) dataclass from a YAML path or a dict.

    Keys absent from ``source`` fall back to the dataclass defaults.
    """
    if isinstance(source, dict):
        data = source
    else:
        with open(source, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    return _from_dict(config_cls, data)


def config_to_dict(config: Any) -> Dict[str, Any]:
    """Recursively convert a dataclass config to a plain dict."""
    return dataclasses.asdict(config)


def dump_config(config: Any, path: str) -> None:
    """Serialise a dataclass config to a YAML file."""
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config_to_dict(config), handle, sort_keys=False)


def merge_overrides(data: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge ``overrides`` into ``data`` without mutating either."""
    merged = dict(data)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_overrides(merged[key], value)
        else:
            merged[key] = value
    return merged
