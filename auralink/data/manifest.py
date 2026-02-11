"""JSONL manifests: one audio clip with its captions and/or event labels per line."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class ManifestItem:
    """A single dataset entry.

    ``captions`` is used by the captioner; ``labels`` (event names) by the
    tagger. Either may be empty depending on the task.
    """

    audio_path: str
    captions: List[str] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    duration: Optional[float] = None
    uid: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ManifestItem":
        return cls(
            audio_path=data["audio_path"],
            captions=list(data.get("captions", [])),
            labels=list(data.get("labels", [])),
            duration=data.get("duration"),
            uid=data.get("uid"),
        )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return {k: v for k, v in data.items() if v is not None}


def read_manifest(path: str) -> List[ManifestItem]:
    """Read a ``.jsonl`` manifest into a list of :class:`ManifestItem`."""
    items: List[ManifestItem] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                items.append(ManifestItem.from_dict(json.loads(line)))
    return items


def write_manifest(path: str, items: Iterable[ManifestItem]) -> None:
    """Write manifest items to ``path`` as one JSON object per line."""
    with open(path, "w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
