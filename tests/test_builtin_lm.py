import torch

from auralink.llm.builtin_lm import TinyDecoderLM
from auralink.llm.tokenizer import CharTokenizer
from auralink.utils.seed import set_seed


def _lm(tok, **kw):
    return TinyDecoderLM(vocab_size=tok.vocab_size, pad_id=tok.pad_id, **kw)


def test_forward_logits_shape():
    tok = CharTokenizer.default()
    lm = _lm(tok, d_model=64, n_layers=2)
    ids = torch.tensor([tok.encode("hi")])
    logits = lm(input_ids=ids)
    assert logits.shape == (1, ids.shape[1], tok.vocab_size)


def test_inputs_embeds_path_matches_ids():
    tok = CharTokenizer.default()
    lm = _lm(tok, d_model=32).eval()
    ids = torch.tensor([tok.encode("ab")])
    with torch.no_grad():
        from_ids = lm(input_ids=ids)
        from_embeds = lm(inputs_embeds=lm.embed_tokens(ids))
    assert torch.allclose(from_ids, from_embeds, atol=1e-5)


def test_attention_is_causal():
    # Changing a future token must not change logits at earlier positions.
    set_seed(0)
    tok = CharTokenizer.default()
    lm = _lm(tok, d_model=32).eval()
    a = torch.tensor([tok.encode("abcd", add_eos=False)])
    b = a.clone()
    b[0, -1] = tok.stoi["z"]
    with torch.no_grad():
        la = lm(input_ids=a)
        lb = lm(input_ids=b)
    assert torch.allclose(la[:, :-1], lb[:, :-1], atol=1e-5)


def test_determinism_in_eval():
    set_seed(1)
    tok = CharTokenizer.default()
    lm = _lm(tok, d_model=32).eval()
    ids = torch.tensor([tok.encode("abc")])
    with torch.no_grad():
        assert torch.allclose(lm(input_ids=ids), lm(input_ids=ids))
