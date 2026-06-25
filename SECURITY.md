# Security Policy

## Supported versions

auralink is pre-1.0; security fixes land on the latest released minor version.

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅        |

## Reporting a vulnerability

Please report suspected vulnerabilities privately via GitHub's
[security advisories](https://github.com/jamesa94/auralink/security/advisories/new)
rather than opening a public issue.

Include enough detail to reproduce the problem (a minimal snippet using the
built-in tiny model is ideal). You can expect an initial response within a few
days. Once a fix is available it will be released and the advisory published.

## Scope notes

auralink loads checkpoints with `torch.load`. Only load checkpoints from sources
you trust — deserialising an untrusted checkpoint can execute arbitrary code.
The optional `HFCausalLM` downloads models from the Hugging Face Hub; the same
trust considerations apply.
