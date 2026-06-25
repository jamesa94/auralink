import torch

from auralink.decoding.beam import beam_search
from auralink.decoding.greedy import greedy_decode
from auralink.decoding.sampling import sample_decode
from auralink.llm.dummy import TinyDecoderLM
from auralink.llm.tokenizer import CharTokenizer
from auralink.utils.seed import set_seed


def _setup(batch=2, q_len=4, dim=32):
    tok = CharTokenizer.default()
    lm = TinyDecoderLM(vocab_size=tok.vocab_size, d_model=dim, n_layers=1, pad_id=tok.pad_id).eval()
    prefix = torch.randn(batch, q_len, dim)
    mask = torch.ones(batch, q_len, dtype=torch.bool)
    return tok, lm, prefix, mask


def test_greedy_shape_and_determinism():
    set_seed(0)
    tok, lm, prefix, mask = _setup()
    a = greedy_decode(lm, prefix, mask, tok.bos_id, tok.eos_id, max_new_tokens=10)
    b = greedy_decode(lm, prefix, mask, tok.bos_id, tok.eos_id, max_new_tokens=10)
    assert a.shape[0] == 2
    assert a.shape[1] <= 10
    assert torch.equal(a, b)


def test_beam_search_shape():
    set_seed(0)
    tok, lm, prefix, mask = _setup()
    out = beam_search(lm, prefix, mask, tok.bos_id, tok.eos_id, max_new_tokens=8, beam_size=3)
    assert out.shape[0] == 2
    assert torch.isfinite(out.float()).all()


def test_sampling_is_reproducible_with_generator():
    set_seed(0)
    tok, lm, prefix, mask = _setup()
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    a = sample_decode(lm, prefix, mask, tok.bos_id, tok.eos_id, max_new_tokens=8, generator=g1)
    b = sample_decode(lm, prefix, mask, tok.bos_id, tok.eos_id, max_new_tokens=8, generator=g2)
    assert torch.equal(a, b)


def test_top_k_and_top_p_run():
    set_seed(0)
    tok, lm, prefix, mask = _setup()
    out = sample_decode(
        lm, prefix, mask, tok.bos_id, tok.eos_id, max_new_tokens=6, top_k=5, top_p=0.9
    )
    assert out.shape[0] == 2
