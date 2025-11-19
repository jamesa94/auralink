"""A tiny name -> class registry so encoders can be built from config."""

from __future__ import annotations

from typing import Callable, Dict, List, Type

from auralink.encoders.base import AudioEncoder

_ENCODERS: Dict[str, Type[AudioEncoder]] = {}


def register_encoder(name: str) -> Callable[[Type[AudioEncoder]], Type[AudioEncoder]]:
    """Class decorator that registers an :class:`AudioEncoder` under ``name``."""

    def decorator(cls: Type[AudioEncoder]) -> Type[AudioEncoder]:
        if name in _ENCODERS:
            raise ValueError(f"encoder '{name}' is already registered")
        _ENCODERS[name] = cls
        return cls

    return decorator


def build_encoder(name: str, **kwargs: object) -> AudioEncoder:
    """Instantiate a registered encoder by name."""
    if name not in _ENCODERS:
        raise KeyError(f"unknown encoder '{name}'; available: {list_encoders()}")
    return _ENCODERS[name](**kwargs)


def list_encoders() -> List[str]:
    """Return the sorted names of all registered encoders."""
    return sorted(_ENCODERS)
