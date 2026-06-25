# API reference

The curated public surface. Everything here is importable from the top-level
`auralink` package unless noted.

## Models

### `build_captioner(config: CaptionerConfig, tokenizer: CharTokenizer) -> AudioCaptioner`
Assemble a captioner from a config. The connector is always projected into the
LM hidden size.

### `AudioCaptioner`
- `forward(waveform, input_ids, attention_mask=None, wav_lengths=None) -> dict`
  with keys `logits`, `text_logits`, `loss`.
- `generate(waveform, decode_config=None, wav_lengths=None) -> list[str]`
- `training_step(batch) -> Tensor`

### `build_tagger(config: TaggerConfig) -> SoundEventTagger`

### `SoundEventTagger`
- `forward(waveform, labels=None, wav_lengths=None) -> dict` (`logits`, `loss`)
- `predict(waveform, threshold=0.5) -> (probs, preds)`

## Components

| Symbol | Module | Purpose |
|--------|--------|---------|
| `LogMelFrontend` | `auralink.audio` | log-mel features |
| `ConvTransformerEncoder` | `auralink.encoders` | conv-stem + transformer encoder |
| `LinearConnector` / `PoolingConnector` / `QFormerConnector` | `auralink.bridge` | audio→LM bridges |
| `TinyDecoderLM` | `auralink.llm` | built-in offline LM |
| `HFCausalLM` | `auralink.llm` | Hugging Face wrapper (extra: `hf`) |
| `CharTokenizer` | `auralink.llm` | character tokenizer |

Registries: `build_encoder` / `list_encoders`, `build_connector` /
`list_connectors`, `build_language_model` / `list_language_models`. Register
your own with the matching `register_*` decorator.

## Decoding

### `DecodeConfig`
Fields: `strategy` (`"greedy"|"beam"|"sample"`), `max_new_tokens`, `beam_size`,
`temperature`, `top_k`, `top_p`, `length_penalty`.

## Data

- `ManifestItem`, `read_manifest`, `write_manifest`
- `CaptionDataset`, `TaggingDataset`
- `collate_captions`, `collate_tags`, `CaptionCollator`
- `SoundEventOntology` — `default()`, `from_file(path)`, `encode(names)`, `decode(vec)`

## Training

- `Trainer(model, optimizer, scheduler=None, device="cpu", grad_clip=None)`
  with `train_epoch`, `evaluate`, `fit`.
- `get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.0)`
- `caption_lm_loss`, `tagging_bce_loss`

## Metrics

- `evaluate_captions(candidates, references_list, max_n=4) -> dict`
- `evaluate_tags(scores, targets, threshold=0.5) -> dict`
- `sentence_bleu`, `corpus_bleu`, `rouge_l`, `corpus_rouge_l`, `compute_cider`
- `mean_average_precision`, `average_precision`, `precision_recall_f1`

## Inference

- `CaptionPipeline(model, sample_rate=16000, decode_config=None)`
- `TagPipeline(model, ontology, sample_rate=16000, threshold=0.5)`

## Utilities

- `set_seed`, `get_logger`, `lengths_to_mask`, `masked_mean`
- `auralink.utils.checkpoint`: `save_checkpoint`, `load_checkpoint`, `load_state_into`
