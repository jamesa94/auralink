# auralink

Bridge audio encoders to language models for **audio captioning** and
**sound-event understanding** — in pure PyTorch, offline by default.

auralink wires three swappable pieces together:

```
waveform → log-mel frontend → audio encoder → connector → language model → text
                                                  │
                                                  └── pooled → classifier → sound-event tags
```

The connector ("bridge") turns the encoder's frame sequence into a short
sequence of prefix tokens that live in the language model's embedding space,
so the LM can be conditioned on audio the same way it is conditioned on text.

A tiny built-in language model ships with the package, which means the whole
pipeline — encode, bridge, generate, train, evaluate — runs on CPU without
downloading a single weight. Swap in a Hugging Face LLM when you are ready to
scale up.

## Install

```bash
pip install auralink                # core (torch, numpy, pyyaml)
pip install "auralink[hf]"          # + transformers, for real LLMs
pip install "auralink[audio]"       # + soundfile, to read audio files
```

From source:

```bash
git clone https://github.com/jamesa94/auralink
cd auralink
pip install -e ".[dev]"
```

## Quickstart

### Caption a clip

```python
import torch
from auralink import CharTokenizer, build_captioner, tiny_captioner_config, DecodeConfig

tokenizer = CharTokenizer.default()
model = build_captioner(tiny_captioner_config(), tokenizer)

waveform = torch.randn(16000)            # 1 second of mono audio at 16 kHz
caption = model.generate(waveform.unsqueeze(0), DecodeConfig(strategy="beam", beam_size=4))
print(caption)                           # untrained model → train it first (see below)
```

### Tag sound events

```python
from auralink import SoundEventOntology, build_tagger, tiny_tagger_config
from auralink import TagPipeline

ontology = SoundEventOntology.default()
tagger = build_tagger(tiny_tagger_config(ontology.num_classes))
events = TagPipeline(tagger, ontology, threshold=0.0)(waveform, top_k=3)
print(events)                            # [('vehicle', 0.61), ('engine', 0.55), ...]
```

### Choose a connector

Three bridges trade off compression vs. fidelity:

| Connector | Prefix tokens | Idea |
|-----------|---------------|------|
| `linear`  | proportional to length | stack adjacent frames, project with SwiGLU/RMSNorm |
| `pooling` | fixed | learned queries attend once over the audio |
| `qformer` | fixed | learned queries refined across cross-attention blocks |

```python
from auralink import build_captioner, tiny_captioner_config
cfg = tiny_captioner_config()
cfg.connector.name = "qformer"
cfg.connector.num_query_tokens = 32
model = build_captioner(cfg, tokenizer)
```

## Train

```python
import torch
from torch.utils.data import DataLoader
from auralink.data import CaptionDataset, CaptionCollator, ManifestItem
from auralink.training import Trainer

items = [ManifestItem(audio_path="clip.wav", captions=["a dog is barking"])]
dataset = CaptionDataset(items, tokenizer, sample_rate=16000)
loader = DataLoader(dataset, batch_size=8, collate_fn=CaptionCollator(tokenizer.pad_id))

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
Trainer(model, optimizer).fit(loader, epochs=5)
```

## Evaluate

```python
from auralink import evaluate_captions, evaluate_tags

scores = evaluate_captions(
    candidates=["a dog is barking"],
    references_list=[["a dog barks", "dog barking"]],
)
print(scores)        # {'bleu_1': ..., 'bleu_4': ..., 'rouge_l': ..., 'cider': ...}
```

## CLI

```bash
auralink info                          # list encoders / connectors / language models
auralink caption clip.wav --strategy beam
auralink tag clip.wav --top-k 5
```

## Documentation

- [docs/architecture.md](docs/architecture.md) — how the pieces fit together
- [docs/usage.md](docs/usage.md) — task-by-task recipes
- [docs/design-notes.md](docs/design-notes.md) — why it is built this way
- [docs/api-reference.md](docs/api-reference.md) — the public surface

## License

MIT — see [LICENSE](LICENSE).
