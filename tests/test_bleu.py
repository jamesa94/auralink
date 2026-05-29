from auralink.metrics.bleu import corpus_bleu, sentence_bleu


def test_identical_is_one():
    score = sentence_bleu("a dog is barking loudly", ["a dog is barking loudly"], n=4)
    assert score > 0.99


def test_disjoint_is_low():
    score = sentence_bleu(
        "a dog is barking loudly", ["totally different unrelated phrase here"], n=4
    )
    assert score < 0.2


def test_partial_overlap_between_zero_and_one():
    score = sentence_bleu("a dog is barking", ["a dog is sleeping"], n=2)
    assert 0.0 < score < 1.0


def test_corpus_bleu_in_range():
    candidates = ["a cat sat down", "the dog ran fast"]
    references = [["a cat sat down"], ["the dog ran fast"]]
    assert 0.0 <= corpus_bleu(candidates, references, n=2) <= 1.0
