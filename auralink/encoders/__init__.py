"""Audio encoders that turn log-mel features into a contextual sequence."""

from auralink.encoders.base import AudioEncoder
from auralink.encoders.conv_transformer import ConvTransformerEncoder
from auralink.encoders.pooling import AttentivePool, MeanPool
from auralink.encoders.registry import build_encoder, list_encoders, register_encoder

__all__ = [
    "AudioEncoder",
    "ConvTransformerEncoder",
    "AttentivePool",
    "MeanPool",
    "build_encoder",
    "list_encoders",
    "register_encoder",
]
