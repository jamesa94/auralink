# Custom encoder plugin

Shows how to add your own encoder: subclass `AudioEncoder`, decorate it with
`@register_encoder("name")`, and select it by setting `config.encoder.name`.

```bash
python examples/custom_encoder/run.py
```

The same pattern works for connectors (`register_connector`) and language
models (`register_language_model`).
