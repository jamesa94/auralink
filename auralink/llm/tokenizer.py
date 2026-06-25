"""A minimal character tokenizer used by the built-in language model.

It is deliberately tiny and dependency-free so the whole captioning stack runs
offline. For real models, swap in the tokenizer that ships with the LLM.
"""

from __future__ import annotations

import string
from typing import Dict, Iterable, List, Sequence


class CharTokenizer:
    PAD = "<pad>"
    BOS = "<bos>"
    EOS = "<eos>"
    UNK = "<unk>"

    def __init__(self, chars: Sequence[str]) -> None:
        specials = [self.PAD, self.BOS, self.EOS, self.UNK]
        self.itos: List[str] = specials + list(chars)
        self.stoi: Dict[str, int] = {tok: i for i, tok in enumerate(self.itos)}
        self.pad_id = self.stoi[self.PAD]
        self.bos_id = self.stoi[self.BOS]
        self.eos_id = self.stoi[self.EOS]
        self.unk_id = self.stoi[self.UNK]

    @classmethod
    def default(cls) -> "CharTokenizer":
        """A sensible lower-case English vocabulary for captions."""
        chars = list(" " + string.ascii_lowercase + string.digits + ".,!?'-:;")
        return cls(chars)

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> "CharTokenizer":
        """Build a vocabulary from the unique characters seen in ``texts``."""
        seen: set = set()
        for text in texts:
            seen.update(text.lower())
        return cls(sorted(seen))

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        ids: List[int] = [self.bos_id] if add_bos else []
        ids.extend(self.stoi.get(ch, self.unk_id) for ch in text.lower())
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        specials = {self.pad_id, self.bos_id, self.eos_id, self.unk_id}
        chars: List[str] = []
        for raw in ids:
            idx = int(raw)
            if idx == self.eos_id:
                break
            if skip_special and idx in specials:
                continue
            if 0 <= idx < len(self.itos):
                chars.append(self.itos[idx])
        return "".join(chars)
