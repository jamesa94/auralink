"""A pure-torch log-mel spectrogram frontend."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from auralink.audio.melscale import mel_filterbank


class LogMelFrontend(nn.Module):
    """Turn a raw waveform into a log-mel spectrogram of shape ``[B, T, n_mels]``.

    The mel filterbank and analysis window are stored as non-persistent buffers
    so checkpoints stay small and the module moves cleanly across devices.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: Optional[int] = None,
        n_mels: int = 64,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        power: float = 2.0,
        log_offset: float = 1e-6,
        center: bool = True,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.n_mels = n_mels
        self.power = power
        self.log_offset = log_offset
        self.center = center

        fb = mel_filterbank(n_mels, n_fft, sample_rate, f_min, f_max)
        self.register_buffer("mel_fb", fb, persistent=False)
        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)

    @property
    def output_dim(self) -> int:
        return self.n_mels

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(waveform.device),
            center=self.center,
            return_complex=True,
        )  # [B, F, T]
        power_spec = spec.abs().pow(self.power)
        mel = torch.matmul(self.mel_fb.to(waveform.device), power_spec)  # [B, n_mels, T]
        log_mel = torch.log(mel + self.log_offset)
        return log_mel.transpose(1, 2).contiguous()  # [B, T, n_mels]
