from auralink.cli import main


def test_info_lists_components(capsys):
    code = main(["info"])
    assert code == 0
    out = capsys.readouterr().out
    assert "auralink" in out
    assert "conv_transformer" in out
    assert "linear" in out


def test_caption_with_random_model_runs(capsys):
    code = main(["caption", "does-not-exist.wav", "--max-new-tokens", "4"])
    assert code == 0


def test_tag_with_random_model_runs(capsys):
    code = main(["tag", "does-not-exist.wav", "--top-k", "3"])
    assert code == 0
    out = capsys.readouterr().out
    assert len(out.strip().splitlines()) == 3


def test_no_command_prints_help(capsys):
    assert main([]) == 1
