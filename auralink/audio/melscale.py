"""Mel-scale conversions and a triangular mel filterbank (HTK convention)."""

from __future__ import annotations

import math
from typing import Optional

import torch


def hz_to_mel(freq: float) -> float:
    """Convert a frequency in Hz to the HTK mel scale."""
    return 2595.0 * math.log10(1.0 + freq / 700.0)


def mel_to_hz(mel: float) -> float:
    """Convert a value on the HTK mel scale back to Hz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(
    n_mels: int,
    n_fft: int,
    sample_rate: int,
    f_min: float = 0.0,
    f_max: Optional[float] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Construct a ``[n_mels, n_fft // 2 + 1]`` triangular mel filterbank.

    The filters are normalised to peak at 1.0 (no area normalisation), which
    keeps the log-mel features in a comfortable numeric range for small models.
    """
    if f_max is None:
        f_max = sample_rate / 2.0
    n_freqs = n_fft // 2 + 1
    all_freqs = torch.linspace(0.0, sample_rate / 2.0, n_freqs, dtype=dtype)

    m_min, m_max = hz_to_mel(f_min), hz_to_mel(f_max)
    m_points = torch.linspace(m_min, m_max, n_mels + 2, dtype=dtype)
    f_points = 700.0 * (10.0 ** (m_points / 2595.0) - 1.0)

    fb = torch.zeros(n_mels, n_freqs, dtype=dtype)
    for m in range(1, n_mels + 1):
        f_left, f_center, f_right = f_points[m - 1], f_points[m], f_points[m + 1]
        left_slope = (all_freqs - f_left) / torch.clamp(f_center - f_left, min=1e-8)
        right_slope = (f_right - all_freqs) / torch.clamp(f_right - f_center, min=1e-8)
        fb[m - 1] = torch.clamp(torch.minimum(left_slope, right_slope), min=0.0)
    return fb
