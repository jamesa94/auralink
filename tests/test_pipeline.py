from auralink.audio.io import synthetic_waveform
from auralink.config.presets import tiny_captioner_config, tiny_tagger_config
from auralink.data.ontology import SoundEventOntology
from auralink.decoding.base import DecodeConfig
from auralink.inference.pipeline import CaptionPipeline, TagPipeline
from auralink.llm.tokenizer import CharTokenizer
from auralink.model.factory import build_captioner, build_tagger
from auralink.utils.seed import set_seed


def test_caption_pipeline_returns_string():
    set_seed(0)
    tok = CharTokenizer.default()
    model = build_captioner(tiny_captioner_config(), tok)
    pipe = CaptionPipeline(model, decode_config=DecodeConfig(max_new_tokens=8))
    out = pipe(synthetic_waveform(16000, seed=0))
    assert isinstance(out, str)


def test_tag_pipeline_returns_scored_labels():
    set_seed(0)
    onto = SoundEventOntology.default()
    model = build_tagger(tiny_tagger_config(onto.num_classes))
    pipe = TagPipeline(model, onto, threshold=0.0)
    out = pipe(synthetic_waveform(16000, seed=0), top_k=3)
    assert len(out) == 3
    assert all(isinstance(name, str) and isinstance(score, float) for name, score in out)
