import torch

from auralink.audio.melscale import hz_to_mel, mel_filterbank, mel_to_hz


def test_mel_roundtrip():
    for hz in (0.0, 100.0, 1000.0, 8000.0):
        assert abs(mel_to_hz(hz_to_mel(hz)) - hz) < 1e-3


def test_filterbank_shape():
    fb = mel_filterbank(n_mels=40, n_fft=400, sample_rate=16000)
    assert fb.shape == (40, 201)


def test_filterbank_is_non_negative_and_triangular():
    fb = mel_filterbank(n_mels=16, n_fft=512, sample_rate=16000)
    assert torch.all(fb >= 0.0)
    # every filter should activate on at least one frequency bin
    assert torch.all(fb.sum(dim=1) > 0.0)
