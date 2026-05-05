# Caption a clip

Runs the full caption path (frontend → encoder → connector → LM → beam search)
with the built-in tiny model. No weights are downloaded.

```bash
python examples/caption_a_clip/run.py
```

The model is randomly initialised, so the caption is meaningless — swap in a
trained checkpoint or a real LLM for sensible text.
