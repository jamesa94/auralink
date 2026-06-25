"""Autoregressive decoding strategies for the captioner."""

from auralink.decoding.base import DecodeConfig, next_token_logits
from auralink.decoding.beam import beam_search
from auralink.decoding.greedy import greedy_decode
from auralink.decoding.sampling import sample_decode

__all__ = [
    "DecodeConfig",
    "beam_search",
    "greedy_decode",
    "next_token_logits",
    "sample_decode",
]
