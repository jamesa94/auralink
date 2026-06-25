# Contributing

Thanks for considering a contribution to auralink. The project values small,
well-tested changes that keep the offline-by-default story intact.

## Development setup

```bash
git clone https://github.com/jamesa94/auralink
cd auralink
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pre-commit install
```

## Before opening a PR

Run the same checks CI runs:

```bash
ruff check .
ruff format --check .
mypy
pytest
```

New tests should stay CPU-only and deterministic (seed with
`auralink.utils.seed.set_seed`) and must not download weights or data — use the
built-in `TinyDecoderLM` and synthetic audio.

## Guidelines

- Keep changes focused; one concern per pull request.
- Add or update tests for any behaviour change.
- Add a type annotation for new public functions; `mypy` runs in CI.
- Update `docs/` and `CHANGELOG.md` for user-facing changes.
- Adding an encoder, connector or language model? Register it and add a short
  test, following the existing ones in `tests/`.

## Commit messages

Conventional-commit prefixes (`feat:`, `fix:`, `docs:`, `refactor:`, `chore:`,
`test:`) are appreciated but not required.
