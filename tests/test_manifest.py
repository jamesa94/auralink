from auralink.data.manifest import ManifestItem, read_manifest, write_manifest


def test_manifest_roundtrip(tmp_path):
    items = [
        ManifestItem(audio_path="a.wav", captions=["a dog"], labels=["dog"], uid="x1"),
        ManifestItem(audio_path="b.wav", captions=["rain"], labels=["rain"], duration=2.0),
    ]
    path = tmp_path / "manifest.jsonl"
    write_manifest(str(path), items)
    loaded = read_manifest(str(path))
    assert len(loaded) == 2
    assert loaded[0].audio_path == "a.wav"
    assert loaded[0].captions == ["a dog"]
    assert loaded[1].duration == 2.0


def test_from_dict_defaults():
    item = ManifestItem.from_dict({"audio_path": "x.wav", "captions": ["c"]})
    assert item.labels == []
    assert item.duration is None
