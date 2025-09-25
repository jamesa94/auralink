import auralink


def test_version_is_a_string():
    assert isinstance(auralink.__version__, str)
    assert auralink.__version__.count(".") >= 2
