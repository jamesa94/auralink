"""Tag sound events in a clip with the built-in tagger."""

from auralink import SoundEventOntology, TagPipeline, build_tagger, tiny_tagger_config
from auralink.audio.io import synthetic_waveform


def main() -> None:
    ontology = SoundEventOntology.default()
    tagger = build_tagger(tiny_tagger_config(ontology.num_classes))

    waveform = synthetic_waveform(16000, freq=220.0, seed=1)
    pipeline = TagPipeline(tagger, ontology, threshold=0.0)
    for name, score in pipeline(waveform, top_k=5):
        print(f"{score:.3f}  {name}")


if __name__ == "__main__":
    main()
