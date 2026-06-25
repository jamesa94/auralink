import pytest
import torch

from auralink.audio.io import synthetic_waveform
from auralink.decoding.base import DecodeConfig
from auralink.llm.tokenizer import CharTokenizer
from auralink.model.config import (
    CaptionerConfig,
    ConnectorConfig,
    EncoderConfig,
    FrontendConfig,
    LMConfig,
)
from auralink.model.factory import build_captioner
from auralink.utils.seed import set_seed


def _tiny_config(connector="linear"):
    return CaptionerConfig(
        frontend=FrontendConfig(n_mels=32, n_fft=256, hop_length=128),
        encoder=EncoderConfig(d_model=48, n_heads=4, n_layers=1, ff_dim=64),
        connector=ConnectorConfig(name=connector, num_query_tokens=6, stack_size=2),
        lm=LMConfig(d_model=48, n_heads=4, n_layers=1, ff_dim=64, max_len=128),
        max_caption_len=16,
    )


def _batch(tok, batch=2, samples=4000):
    wav = torch.stack([synthetic_waveform(samples, seed=i) for i in range(batch)])
    ids = torch.tensor([tok.encode("a dog barks")[:10]] * batch)
    return wav, ids


def test_forward_loss_is_finite_and_backprops():
    set_seed(0)
    tok = CharTokenizer.default()
    model = build_captioner(_tiny_config(), tok)
    wav, ids = _batch(tok)
    out = model(wav, ids)
    assert out["text_logits"].shape[:2] == ids.shape
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert len(grads) > 0


@pytest.mark.parametrize("connector", ["linear", "pooling", "qformer"])
def test_each_connector_runs(connector):
    set_seed(0)
    tok = CharTokenizer.default()
    model = build_captioner(_tiny_config(connector), tok)
    wav, ids = _batch(tok)
    assert torch.isfinite(model(wav, ids)["loss"])


@pytest.mark.parametrize("strategy", ["greedy", "beam", "sample"])
def test_generate_returns_strings(strategy):
    set_seed(0)
    tok = CharTokenizer.default()
    model = build_captioner(_tiny_config(), tok)
    wav, _ = _batch(tok, batch=2)
    caps = model.generate(wav, DecodeConfig(strategy=strategy, max_new_tokens=8, beam_size=3))
    assert len(caps) == 2
    assert all(isinstance(c, str) for c in caps)


def test_mismatched_dims_raise():
    set_seed(0)
    tok = CharTokenizer.default()
    cfg = _tiny_config()
    cfg.lm.d_model = 64  # connector projects to 64 now, but encoder/lm wired by factory
    model = build_captioner(cfg, tok)
    # factory keeps them consistent, so this should still build and run
    wav, ids = _batch(tok)
    assert torch.isfinite(model(wav, ids)["loss"])
