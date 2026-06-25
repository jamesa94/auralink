from auralink.metrics.cider import compute_cider


def test_identical_captions_score_positive():
    candidates = ["a dog barking loudly", "rain falling on the roof"]
    references = [["a dog barking loudly"], ["rain falling on the roof"]]
    mean, scores = compute_cider(candidates, references)
    assert mean > 0.0
    assert all(s >= 0.0 for s in scores)


def test_identical_beats_unrelated():
    candidates = ["a cat meows softly", "a cat meows softly"]
    identical = [["a cat meows softly"], ["a cat meows softly"]]
    unrelated = [["a truck engine roaring"], ["birds chirping outside"]]
    mean_identical, _ = compute_cider(candidates, identical)
    mean_unrelated, _ = compute_cider(candidates, unrelated)
    assert mean_identical > mean_unrelated


def test_returns_per_sample_scores():
    candidates = ["one two three", "four five six"]
    references = [["one two three"], ["four five six"]]
    _, scores = compute_cider(candidates, references)
    assert len(scores) == 2
