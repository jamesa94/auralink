"""Ready-made model configs for quick experiments."""

from __future__ import annotations

from typing import Any, Callable, Dict

from auralink.model.config import (
    CaptionerConfig,
    ConnectorConfig,
    EncoderConfig,
    FrontendConfig,
    LMConfig,
    TaggerConfig,
)


def tiny_captioner_config() -> CaptionerConfig:
    """A small captioner that trains comfortably on CPU."""
    return CaptionerConfig(
        frontend=FrontendConfig(n_mels=40, n_fft=400, hop_length=160),
        encoder=EncoderConfig(d_model=128, n_heads=4, n_layers=2, ff_dim=256),
        connector=ConnectorConfig(name="linear", stack_size=2),
        lm=LMConfig(d_model=128, n_heads=4, n_layers=2, ff_dim=256, max_len=256),
        max_caption_len=64,
    )


def base_captioner_config() -> CaptionerConfig:
    """A larger captioner with a Q-Former connector."""
    return CaptionerConfig(
        frontend=FrontendConfig(n_mels=64),
        encoder=EncoderConfig(d_model=256, n_heads=8, n_layers=4, ff_dim=1024),
        connector=ConnectorConfig(name="qformer", num_query_tokens=32, n_heads=8, n_layers=2),
        lm=LMConfig(d_model=256, n_heads=8, n_layers=4, ff_dim=1024, max_len=512),
        max_caption_len=128,
    )


def tiny_tagger_config(num_classes: int = 12) -> TaggerConfig:
    """A small multi-label tagger."""
    return TaggerConfig(
        frontend=FrontendConfig(n_mels=40),
        encoder=EncoderConfig(d_model=128, n_heads=4, n_layers=2, ff_dim=256),
        num_classes=num_classes,
        pool="attentive",
        hidden_dim=128,
    )


PRESETS: Dict[str, Callable[..., Any]] = {
    "tiny_captioner": tiny_captioner_config,
    "base_captioner": base_captioner_config,
    "tiny_tagger": tiny_tagger_config,
}


def get_preset(name: str, **kwargs: Any) -> Any:
    """Look up a preset config builder by name."""
    if name not in PRESETS:
        raise KeyError(f"unknown preset '{name}'; available: {sorted(PRESETS)}")
    return PRESETS[name](**kwargs)
