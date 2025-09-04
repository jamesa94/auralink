"""auralink: bridge audio encoders to language models.

This package wires a log-mel frontend and an audio encoder to a (small,
built-in or external) language model so it can caption audio clips and
tag sound events.
"""

from auralink._version import __version__

__all__ = ["__version__"]
