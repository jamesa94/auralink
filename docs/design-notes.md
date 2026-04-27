# Design notes

A few opinions baked into the library, and the reasoning behind them.

## Offline by default

The hardest part of reproducing audio-LM work is usually getting the weights and
data to line up. auralink ships a tiny built-in language model
(`TinyDecoderLM`) and a synthetic-audio fallback so the entire stack —
encode, bridge, generate, train, evaluate — runs on a laptop with no network.
Real encoders and LLMs are opt-in, not required. This keeps the test suite fast
and deterministic and makes the examples genuinely runnable.

## The connector is the interesting part

Captioning quality is dominated by how the audio is presented to the LM, not by
the LM itself. So the bridge is a first-class, swappable component with three
implementations spanning the usual trade-off:

- length-proportional tokens (`linear`) keep temporal detail but grow the
  sequence;
- fixed-width tokens (`pooling`, `qformer`) bound the cost regardless of clip
  length.

Borrowing from Gazelle and BLIP-2, the connectors use temporal stacking,
SwiGLU + RMSNorm projections, and learned-query cross-attention rather than a
single linear layer.

## Prefix tokens, not special-token splicing

Audio is injected as `inputs_embeds`: the connector output is concatenated in
front of the caption token embeddings and the whole thing is one LM forward
pass. This avoids reserving vocabulary ids for an `<audio>` placeholder and
works with any LM that accepts input embeddings — including frozen HF models.

## Masks follow the frame rate

The conv stem downsamples time, so a frame mask computed from waveform lengths
has to be downsampled too. `ConvTransformerEncoder` recomputes lengths with the
same conv arithmetic and rebuilds the mask, and the `linear` connector does the
same when it stacks frames. Getting this right is what lets variable-length
clips batch cleanly.

## Pure-python metrics

BLEU, ROUGE-L and CIDEr-D are implemented from scratch with no Java or external
jars (unlike the canonical COCO tooling). They are close enough for tracking
training progress and keep the dependency surface to torch + numpy. CIDEr uses
`log(N + 1)` for the reference count so it behaves on tiny corpora instead of
collapsing to zero.

## Known limitations

- `TinyDecoderLM` is for plumbing, not quality — bring a real LLM for results.
- The linear resampler in `audio.io` is fine for demos; use a sinc resampler
  for production.
- Metrics approximate the reference implementations; they are not a drop-in
  replacement for `aac-metrics` when reporting paper numbers.
