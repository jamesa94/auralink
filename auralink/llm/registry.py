"""Name -> language-model class registry."""

from __future__ import annotations

from typing import Callable, Dict, List, Type

from auralink.llm.base import LanguageModel

_LANGUAGE_MODELS: Dict[str, Type[LanguageModel]] = {}


def register_language_model(
    name: str,
) -> Callable[[Type[LanguageModel]], Type[LanguageModel]]:
    """Class decorator registering a :class:`LanguageModel` under ``name``."""

    def decorator(cls: Type[LanguageModel]) -> Type[LanguageModel]:
        if name in _LANGUAGE_MODELS:
            raise ValueError(f"language model '{name}' is already registered")
        _LANGUAGE_MODELS[name] = cls
        return cls

    return decorator


def build_language_model(name: str, **kwargs: object) -> LanguageModel:
    """Instantiate a registered language model by name."""
    if name not in _LANGUAGE_MODELS:
        raise KeyError(f"unknown language model '{name}'; available: {list_language_models()}")
    return _LANGUAGE_MODELS[name](**kwargs)


def list_language_models() -> List[str]:
    """Return the sorted names of all registered language models."""
    return sorted(_LANGUAGE_MODELS)
