# Architecture

auralink is organised as a small stack of swappable components. Each stage has
a narrow contract so pieces can be replaced without touching the rest.

```
            ┌──────────────┐   ┌───────────────┐   ┌───────────┐   ┌──────────────┐
 waveform → │ LogMelFront. │ → │ AudioEncoder  │ → │ Connector │ → │ LanguageModel│ → caption
            └──────────────┘   └───────┬───────┘   └───────────┘   └──────────────┘
                                       │
                                       └─ pool → classifier → sound-event tags
```

## Stages

### Frontend — `auralink.audio`
`LogMelFrontend` computes a log-mel spectrogram with a pure-torch STFT and a
triangular mel filterbank (`mel_filterbank`). Output: `[B, T, n_mels]`. The
filterbank and window are non-persistent buffers, so checkpoints stay small.

### Encoder — `auralink.encoders`
An `AudioEncoder` maps `[B, T, n_mels]` to a contextual sequence `[B, T', D]`
and, given a frame mask, returns the downsampled mask. The built-in
`ConvTransformerEncoder` uses a strided conv stem (reducing the frame rate by
`conv_stride ** n_conv`) followed by a norm-first transformer. Encoders are
looked up through a registry (`register_encoder` / `build_encoder`).

### Connector — `auralink.bridge`
A `Connector` turns the encoder sequence into prefix tokens `[B, Q, D_lm]` that
live in the language model's embedding space. Three implementations:

- **LinearConnector** stacks `stack_size` adjacent frames and projects them
  with a SwiGLU block wrapped in RMSNorm. `Q` scales with clip length.
- **PoolingConnector** lets a fixed bank of learned queries attend once over
  the audio, producing a constant `Q`.
- **QFormerConnector** refines the learned queries across several blocks that
  alternate self-attention and audio cross-attention (BLIP-2 style).

### Language model — `auralink.llm`
A `LanguageModel` exposes `embed_tokens` and a `forward` that accepts
`inputs_embeds`. That is the whole trick: the captioner concatenates the audio
prefix with the caption token embeddings and feeds one sequence to the LM.
`TinyDecoderLM` is a small built-in decoder so everything runs offline;
`HFCausalLM` wraps any `transformers` causal LM.

## Composition

`AudioCaptioner` owns `frontend → encoder → connector → lm` and trains with a
next-token objective on the text positions only (the audio prefix is not
supervised). `SoundEventTagger` reuses the frontend and encoder, then pools and
classifies. `auralink.model.factory` builds either from a dataclass config,
always projecting the connector into the LM hidden size so the prefix can be
concatenated.

## Data flow during training

1. `CaptionDataset` yields `{waveform, input_ids, caption}`.
2. `CaptionCollator` pads waveforms and tokens and builds the attention mask.
3. `Trainer` calls `model.training_step(batch)` which returns a scalar loss.
4. Standard optimizer/scheduler step; metrics tracked by `MetricTracker`.

See [design-notes.md](design-notes.md) for the reasoning behind these choices.
