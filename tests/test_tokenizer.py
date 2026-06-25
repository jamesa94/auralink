from auralink.llm.tokenizer import CharTokenizer


def test_roundtrip_default():
    tok = CharTokenizer.default()
    ids = tok.encode("hello world")
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id
    assert tok.decode(ids) == "hello world"


def test_from_texts_vocab_size():
    tok = CharTokenizer.from_texts(["abc", "cab"])
    # 4 specials + {a, b, c}
    assert tok.vocab_size == 7


def test_unknown_char_maps_to_unk():
    tok = CharTokenizer(list("abc"))
    ids = tok.encode("axc", add_bos=False, add_eos=False)
    assert tok.unk_id in ids


def test_decode_stops_at_eos():
    tok = CharTokenizer.default()
    ids = tok.encode("hi", add_bos=False) + tok.encode("ignored", add_bos=False)
    assert tok.decode(ids) == "hi"
