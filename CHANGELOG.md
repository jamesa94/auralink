# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims
to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-06-26

First public release.

### Added
- Pure-torch `LogMelFrontend` with a triangular mel filterbank and SpecAugment.
- `ConvTransformerEncoder` with a strided conv stem and an encoder registry.
- Three connectors bridging audio to LM space: `linear`, `pooling`, `qformer`.
- `TinyDecoderLM` (built-in, offline) and an `HFCausalLM` wrapper for real LLMs.
- `AudioCaptioner` (next-token training + greedy/beam/sampling decoding) and a
  multi-label `SoundEventTagger`.
- JSONL manifests, caption/tagging datasets, padding collates and a sound-event
  ontology.
- Pure-python BLEU, ROUGE-L and CIDEr-D plus mAP / F1 tagging metrics.
- A minimal `Trainer`, warmup-cosine schedule, config dataclasses with YAML
  loading, inference pipelines and an `auralink` CLI.

[Unreleased]: https://github.com/jamesa94/auralink/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jamesa94/auralink/releases/tag/v0.1.0
