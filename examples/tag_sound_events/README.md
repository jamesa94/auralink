# Tag sound events

Runs the multi-label tagger and prints the top-k scored events from the default
ontology.

```bash
python examples/tag_sound_events/run.py
```

Scores come from an untrained model; train the tagger on a labelled manifest for
meaningful predictions.
