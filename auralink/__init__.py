"""auralink: bridge audio encoders to language models.

This package wires a log-mel frontend and an audio encoder to a (small,
built-in or external) language model so it can caption audio clips and tag
sound events. The defaults are offline-first: nothing is downloaded to run the
tiny built-in model end to end.
"""

from auralink._version import __version__
from auralink.audio.features import LogMelFrontend
from auralink.audio.io import load_audio
from auralink.bridge import LinearConnector, PoolingConnector, QFormerConnector
from auralink.config.presets import (
    base_captioner_config,
    tiny_captioner_config,
    tiny_tagger_config,
)
from auralink.data.ontology import SoundEventOntology
from auralink.decoding.base import DecodeConfig
from auralink.encoders import ConvTransformerEncoder
from auralink.inference.pipeline import CaptionPipeline, TagPipeline
from auralink.llm import CharTokenizer, TinyDecoderLM
from auralink.metrics import evaluate_captions, evaluate_tags
from auralink.model import AudioCaptioner, SoundEventTagger, build_captioner, build_tagger
from auralink.training import Trainer

__all__ = [
    "__version__",
    "AudioCaptioner",
    "SoundEventTagger",
    "build_captioner",
    "build_tagger",
    "LogMelFrontend",
    "ConvTransformerEncoder",
    "LinearConnector",
    "PoolingConnector",
    "QFormerConnector",
    "TinyDecoderLM",
    "CharTokenizer",
    "DecodeConfig",
    "CaptionPipeline",
    "TagPipeline",
    "SoundEventOntology",
    "Trainer",
    "load_audio",
    "evaluate_captions",
    "evaluate_tags",
    "tiny_captioner_config",
    "base_captioner_config",
    "tiny_tagger_config",
]
