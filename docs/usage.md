# Usage

All snippets run on CPU with the built-in tiny model — no downloads required.

## Building a model

```python
from auralink import CharTokenizer, build_captioner, tiny_captioner_config

tokenizer = CharTokenizer.default()
model = build_captioner(tiny_captioner_config(), tokenizer)
```

Configs are plain dataclasses. Edit them directly or load from YAML:

```python
from auralink.config import load_config, dump_config
from auralink.model.config import CaptionerConfig

cfg = load_config("captioner.yaml", CaptionerConfig)   # missing keys use defaults
dump_config(cfg, "resolved.yaml")
```

## Datasets and manifests

A manifest is a JSONL file, one clip per line:

```json
{"audio_path": "clips/0001.wav", "captions": ["a dog is barking"], "labels": ["dog"]}
{"audio_path": "clips/0002.wav", "captions": ["rain on a window"], "labels": ["rain"]}
```

```python
from auralink.data import read_manifest, CaptionDataset, CaptionCollator
from torch.utils.data import DataLoader

items = read_manifest("train.jsonl")
dataset = CaptionDataset(items, tokenizer, sample_rate=16000, max_caption_len=64)
loader = DataLoader(dataset, batch_size=16, shuffle=True,
                    collate_fn=CaptionCollator(tokenizer.pad_id))
```

If an audio file is missing, the dataset falls back to a deterministic synthetic
tone so examples and tests stay runnable.

## Training

```python
import torch
from auralink.training import Trainer, get_warmup_cosine_scheduler

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
scheduler = get_warmup_cosine_scheduler(optimizer, warmup_steps=100, total_steps=2000)
trainer = Trainer(model, optimizer, scheduler=scheduler, grad_clip=1.0)
history = trainer.fit(loader, epochs=10, val_dataloader=loader)
```

## Decoding

```python
from auralink import DecodeConfig

greedy = model.generate(waveform_batch, DecodeConfig(strategy="greedy"))
beam   = model.generate(waveform_batch, DecodeConfig(strategy="beam", beam_size=5))
sample = model.generate(waveform_batch, DecodeConfig(strategy="sample", temperature=0.8, top_p=0.9))
```

## Tagging

```python
from auralink import SoundEventOntology, build_tagger, tiny_tagger_config, TagPipeline

ontology = SoundEventOntology.default()                 # or from_file("labels.txt")
tagger = build_tagger(tiny_tagger_config(ontology.num_classes))
scored = TagPipeline(tagger, ontology, threshold=0.3)(waveform)
```

## Evaluation

```python
from auralink import evaluate_captions, evaluate_tags

caption_scores = evaluate_captions(predictions, references_list)
tag_scores = evaluate_tags(prob_matrix, target_matrix, threshold=0.5)
```

## Checkpoints

```python
from auralink.utils.checkpoint import save_checkpoint, load_state_into

save_checkpoint("captioner.pt", model)
load_state_into(model, "captioner.pt")
```

## Using a real LLM

```python
from auralink.llm import HFCausalLM   # requires: pip install "auralink[hf]"

lm = HFCausalLM("gpt2", freeze=True)  # encoder + connector train, LLM stays frozen
```
