from auralink.metrics.rouge import corpus_rouge_l, rouge_l


def test_identical_is_one():
    assert rouge_l("a dog barks", ["a dog barks"]) > 0.99


def test_no_overlap_is_zero():
    assert rouge_l("a dog barks", ["xyz qrs tuv"]) == 0.0


def test_partial_between_zero_and_one():
    value = rouge_l("a dog barks loudly", ["a dog sleeps"])
    assert 0.0 < value < 1.0


def test_corpus_average_in_range():
    value = corpus_rouge_l(["a b c", "d e f"], [["a b c"], ["d x f"]])
    assert 0.0 <= value <= 1.0
