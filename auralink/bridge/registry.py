"""Name -> connector class registry."""

from __future__ import annotations

from typing import Callable, Dict, List, Type

from auralink.bridge.base import Connector

_CONNECTORS: Dict[str, Type[Connector]] = {}


def register_connector(name: str) -> Callable[[Type[Connector]], Type[Connector]]:
    """Class decorator registering a :class:`Connector` under ``name``."""

    def decorator(cls: Type[Connector]) -> Type[Connector]:
        if name in _CONNECTORS:
            raise ValueError(f"connector '{name}' is already registered")
        _CONNECTORS[name] = cls
        return cls

    return decorator


def build_connector(name: str, **kwargs: object) -> Connector:
    """Instantiate a registered connector by name."""
    if name not in _CONNECTORS:
        raise KeyError(f"unknown connector '{name}'; available: {list_connectors()}")
    return _CONNECTORS[name](**kwargs)


def list_connectors() -> List[str]:
    """Return the sorted names of all registered connectors."""
    return sorted(_CONNECTORS)
