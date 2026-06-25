import torch

from auralink.audio.features import LogMelFrontend
from auralink.audio.io import synthetic_waveform


def test_frontend_output_shape():
    frontend = LogMelFrontend(n_mels=64, n_fft=400, hop_length=160)
    wav = synthetic_waveform(16000, seed=0)
    feats = frontend(wav)
    assert feats.dim() == 3
    assert feats.shape[0] == 1
    assert feats.shape[2] == 64
    assert feats.shape[1] > 1


def test_frontend_batched():
    frontend = LogMelFrontend(n_mels=32)
    wav = torch.stack([synthetic_waveform(8000, seed=1), synthetic_waveform(8000, seed=2)])
    feats = frontend(wav)
    assert feats.shape[0] == 2
    assert feats.shape[2] == 32


def test_output_dim_property():
    assert LogMelFrontend(n_mels=48).output_dim == 48
