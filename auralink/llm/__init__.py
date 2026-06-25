"""Language-model interface plus a small built-in model and an HF wrapper."""

from auralink.llm.base import LanguageModel
from auralink.llm.dummy import TinyDecoderLM
from auralink.llm.hf import HFCausalLM
from auralink.llm.registry import (
    build_language_model,
    list_language_models,
    register_language_model,
)
from auralink.llm.tokenizer import CharTokenizer

__all__ = [
    "LanguageModel",
    "TinyDecoderLM",
    "HFCausalLM",
    "CharTokenizer",
    "build_language_model",
    "list_language_models",
    "register_language_model",
]
