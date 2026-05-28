"""Waveform loading, resampling and a synthetic generator for offline demos."""

from __future__ import annotations

import math
from typing import Optional

import torch


def resample(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """Linearly resample a mono waveform. Good enough for demos and tests.

    For production-quality resampling use a polyphase/sinc resampler; this keeps
    the dependency surface to plain torch.
    """
    if orig_sr == target_sr:
        return waveform
    x = waveform.reshape(1, 1, -1)
    new_len = round(x.shape[-1] * target_sr / orig_sr)
    y = torch.nn.functional.interpolate(x, size=new_len, mode="linear", align_corners=False)
    return y.reshape(-1)


def load_audio(path: str, sample_rate: int = 16000, mono: bool = True) -> torch.Tensor:
    """Load an audio file as a 1-D float tensor resampled to ``sample_rate``.

    Requires the optional ``soundfile`` dependency (``pip install auralink[audio]``).
    """
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "load_audio requires the optional 'soundfile' dependency; "
            "install it with `pip install auralink[audio]`"
        ) from exc

    data, sr = sf.read(path, dtype="float32", always_2d=True)  # [num_samples, channels]
    wav = torch.from_numpy(data).transpose(0, 1)  # [channels, num_samples]
    if mono:
        wav = wav.mean(dim=0)
    if sr != sample_rate:
        wav = resample(wav, sr, sample_rate)
    return wav


def synthetic_waveform(
    num_samples: int,
    freq: float = 440.0,
    sample_rate: int = 16000,
    noise: float = 0.01,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Deterministic sine tone with light noise — handy when no audio file exists."""
    t = torch.arange(num_samples, dtype=torch.float32) / sample_rate
    tone = torch.sin(2.0 * math.pi * freq * t)
    if noise > 0.0 and seed is not None:
        gen = torch.Generator().manual_seed(seed)
        tone = tone + noise * torch.randn(num_samples, generator=gen)
    return (0.5 * tone).clamp(-1.0, 1.0)
