"""Save and load model checkpoints with an optional config payload."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def save_checkpoint(
    path: str,
    model: nn.Module,
    config: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Write ``state_dict`` plus an optional ``config`` and ``extra`` payload."""
    payload: Dict[str, Any] = {"state_dict": model.state_dict()}
    if config is not None:
        payload["config"] = config
    if extra is not None:
        payload["extra"] = extra
    torch.save(payload, path)


def load_checkpoint(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    """Load a checkpoint dict written by :func:`save_checkpoint`."""
    return torch.load(path, map_location=map_location, weights_only=False)


def load_state_into(
    model: nn.Module, path: str, map_location: str = "cpu", strict: bool = True
) -> nn.Module:
    """Load weights from a checkpoint into ``model`` in place."""
    checkpoint = load_checkpoint(path, map_location)
    state = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state, strict=strict)
    return model
