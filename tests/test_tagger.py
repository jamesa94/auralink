import torch

from auralink.audio.io import synthetic_waveform
from auralink.model.config import EncoderConfig, FrontendConfig, TaggerConfig
from auralink.model.factory import build_tagger
from auralink.utils.seed import set_seed


def _cfg(pool="attentive"):
    return TaggerConfig(
        frontend=FrontendConfig(n_mels=32, n_fft=256, hop_length=128),
        encoder=EncoderConfig(d_model=48, n_layers=1, ff_dim=64),
        num_classes=8,
        hidden_dim=32,
        pool=pool,
    )


def test_tagger_forward_loss():
    set_seed(0)
    model = build_tagger(_cfg())
    wav = torch.stack([synthetic_waveform(4000, seed=i) for i in range(2)])
    labels = torch.zeros(2, 8)
    labels[0, 1] = 1.0
    labels[1, 3] = 1.0
    out = model(wav, labels=labels)
    assert out["logits"].shape == (2, 8)
    assert torch.isfinite(out["loss"])


def test_tagger_predict_threshold():
    set_seed(0)
    model = build_tagger(_cfg())
    wav = synthetic_waveform(4000, seed=0).unsqueeze(0)
    probs, preds = model.predict(wav, threshold=0.5)
    assert probs.shape == (1, 8)
    assert preds.dtype == torch.bool


def test_tagger_mean_pool_variant():
    set_seed(0)
    model = build_tagger(_cfg(pool="mean"))
    wav = synthetic_waveform(4000, seed=0).unsqueeze(0)
    assert model(wav)["logits"].shape == (1, 8)
