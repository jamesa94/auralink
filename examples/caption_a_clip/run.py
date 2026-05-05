"""Caption a clip with the built-in tiny model.

The model is untrained, so the output is gibberish — the point is to show the
end-to-end path runs offline with no downloads. See ``train_tiny_captioner``
for a version that actually learns something.
"""

from auralink import CharTokenizer, DecodeConfig, build_captioner, tiny_captioner_config
from auralink.audio.io import synthetic_waveform


def main() -> None:
    tokenizer = CharTokenizer.default()
    model = build_captioner(tiny_captioner_config(), tokenizer)

    waveform = synthetic_waveform(16000, freq=330.0, seed=0).unsqueeze(0)
    caption = model.generate(
        waveform, DecodeConfig(strategy="beam", beam_size=4, max_new_tokens=20)
    )
    print("caption:", caption[0])


if __name__ == "__main__":
    main()
