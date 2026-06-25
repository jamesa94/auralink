from auralink.metrics.cider import compute_cider


def test_identical_captions_score_positive():
    candidates = ["a dog barking loudly", "rain falling on the roof"]
    references = [["a dog barking loudly"], ["rain falling on the roof"]]
    mean, scores = compute_cider(candidates, references)
    assert mean > 0.0
    assert all(s >= 0.0 for s in scores)


def test_returns_per_sample_scores():
    candidates = ["one two three", "four five six"]
    references = [["one two three"], ["four five six"]]
    _, scores = compute_cider(candidates, references)
    assert len(scores) == 2
