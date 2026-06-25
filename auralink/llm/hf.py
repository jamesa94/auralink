"""Adapter that lets any Hugging Face causal LM speak the auralink interface."""

from __future__ import annotations

from typing import Optional

import torch

from auralink.llm.base import LanguageModel
from auralink.llm.registry import register_language_model


@register_language_model("hf")
class HFCausalLM(LanguageModel):
    """Wrap a ``transformers`` causal LM (requires ``pip install auralink[hf]``).

    By default the underlying model is frozen, so only the audio encoder and
    connector are trained — the common recipe when bridging to a large LLM.
    """

    def __init__(
        self,
        model_name_or_path: str,
        dtype: Optional[torch.dtype] = None,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "HFCausalLM requires the optional 'transformers' dependency; "
                "install it with `pip install auralink[hf]`"
            ) from exc

        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=dtype)
        self._hidden = int(self.model.config.hidden_size)
        self._vocab = int(self.model.config.vocab_size)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad_(False)

    @property
    def hidden_size(self) -> int:
        return self._hidden

    @property
    def vocab_size(self) -> int:
        return self._vocab

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings()(input_ids)

    def forward(
        self,
        inputs_embeds: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            input_ids=input_ids if inputs_embeds is None else None,
            attention_mask=attention_mask,
        )
        return outputs.logits
