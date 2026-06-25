"""High-level models: the audio captioner and the sound-event tagger."""

from auralink.model.captioner import AudioCaptioner
from auralink.model.config import (
    CaptionerConfig,
    ConnectorConfig,
    EncoderConfig,
    FrontendConfig,
    LMConfig,
    TaggerConfig,
)
from auralink.model.factory import build_captioner, build_tagger
from auralink.model.tagger import SoundEventTagger

__all__ = [
    "AudioCaptioner",
    "CaptionerConfig",
    "ConnectorConfig",
    "EncoderConfig",
    "FrontendConfig",
    "LMConfig",
    "SoundEventTagger",
    "TaggerConfig",
    "build_captioner",
    "build_tagger",
]
