"""Dataclass configs describing how to assemble the models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FrontendConfig:
    sample_rate: int = 16000
    n_fft: int = 400
    hop_length: int = 160
    n_mels: int = 64
    f_min: float = 0.0
    f_max: Optional[float] = None


@dataclass
class EncoderConfig:
    name: str = "conv_transformer"
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    ff_dim: int = 256
    conv_stride: int = 2
    n_conv: int = 2
    dropout: float = 0.1


@dataclass
class ConnectorConfig:
    name: str = "linear"
    num_query_tokens: int = 16
    stack_size: int = 2
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.0


@dataclass
class LMConfig:
    name: str = "tiny"
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    ff_dim: int = 256
    max_len: int = 512
    dropout: float = 0.1
    model_name_or_path: Optional[str] = None


@dataclass
class CaptionerConfig:
    frontend: FrontendConfig = field(default_factory=FrontendConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    connector: ConnectorConfig = field(default_factory=ConnectorConfig)
    lm: LMConfig = field(default_factory=LMConfig)
    max_caption_len: int = 64


@dataclass
class TaggerConfig:
    frontend: FrontendConfig = field(default_factory=FrontendConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    num_classes: int = 10
    pool: str = "attentive"
    hidden_dim: int = 128
    dropout: float = 0.1
