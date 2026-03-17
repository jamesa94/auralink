"""Thin wrappers turning a trained model into a one-call inference helper."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch

from auralink.audio.io import load_audio
from auralink.data.ontology import SoundEventOntology
from auralink.decoding.base import DecodeConfig
from auralink.model.captioner import AudioCaptioner
from auralink.model.tagger import SoundEventTagger

AudioInput = Union[str, torch.Tensor]


class CaptionPipeline:
    """Caption an audio clip given a path or a waveform tensor."""

    def __init__(
        self,
        model: AudioCaptioner,
        sample_rate: int = 16000,
        decode_config: Optional[DecodeConfig] = None,
    ) -> None:
        self.model = model.eval()
        self.sample_rate = sample_rate
        self.decode_config = decode_config or DecodeConfig()

    def _to_waveform(self, audio: AudioInput) -> torch.Tensor:
        wav = load_audio(audio, self.sample_rate) if isinstance(audio, str) else audio
        return wav.unsqueeze(0) if wav.dim() == 1 else wav

    @torch.no_grad()
    def __call__(
        self, audio: AudioInput, decode_config: Optional[DecodeConfig] = None
    ) -> Union[str, List[str]]:
        wav = self._to_waveform(audio)
        captions = self.model.generate(wav, decode_config or self.decode_config)
        return captions[0] if len(captions) == 1 else captions


class TagPipeline:
    """Return scored sound-event labels for an audio clip."""

    def __init__(
        self,
        model: SoundEventTagger,
        ontology: SoundEventOntology,
        sample_rate: int = 16000,
        threshold: float = 0.5,
    ) -> None:
        self.model = model.eval()
        self.ontology = ontology
        self.sample_rate = sample_rate
        self.threshold = threshold

    def _to_waveform(self, audio: AudioInput) -> torch.Tensor:
        wav = load_audio(audio, self.sample_rate) if isinstance(audio, str) else audio
        return wav.unsqueeze(0) if wav.dim() == 1 else wav

    @torch.no_grad()
    def __call__(
        self, audio: AudioInput, top_k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        wav = self._to_waveform(audio)
        probs, _ = self.model.predict(wav, threshold=self.threshold)
        probs = probs[0]
        scored = sorted(
            ((self.ontology.label_name(i), float(probs[i])) for i in range(len(probs))),
            key=lambda pair: pair[1],
            reverse=True,
        )
        if top_k is not None:
            return scored[:top_k]
        return [(name, score) for name, score in scored if score >= self.threshold]
