# Train a tiny captioner

Overfits the built-in captioner on three synthetic clips and prints the loss
dropping, then decodes one of the training clips.

```bash
python examples/train_tiny_captioner/run.py
```

It runs on CPU in a few seconds. The captions are tied to deterministic
synthetic tones, so the model can memorise them — a quick sanity check that the
whole training path is wired correctly.
