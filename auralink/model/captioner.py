"""The audio captioner: frontend -> encoder -> connector -> language model."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from auralink.audio.features import LogMelFrontend
from auralink.bridge.base import Connector
from auralink.decoding.base import DecodeConfig
from auralink.decoding.beam import beam_search
from auralink.decoding.greedy import greedy_decode
from auralink.decoding.sampling import sample_decode
from auralink.encoders.base import AudioEncoder
from auralink.llm.base import LanguageModel
from auralink.llm.tokenizer import CharTokenizer
from auralink.utils.masking import lengths_to_mask


class AudioCaptioner(nn.Module):
    """Condition a language model on audio prefix tokens to generate captions.

    The audio is turned into ``Q`` prefix embeddings by the connector and
    prepended to the caption token embeddings, so the language model sees a
    single sequence ``[audio | text]`` and is trained with a standard next-token
    objective on the text positions.
    """

    def __init__(
        self,
        frontend: LogMelFrontend,
        encoder: AudioEncoder,
        connector: Connector,
        lm: LanguageModel,
        tokenizer: CharTokenizer,
        max_caption_len: int = 64,
    ) -> None:
        super().__init__()
        if connector.output_dim != lm.hidden_size:
            raise ValueError(
                f"connector output_dim ({connector.output_dim}) must match the LM "
                f"hidden size ({lm.hidden_size})"
            )
        self.frontend = frontend
        self.encoder = encoder
        self.connector = connector
        self.lm = lm
        self.tokenizer = tokenizer
        self.max_caption_len = max_caption_len

    def encode_audio(
        self, waveform: torch.Tensor, wav_lengths: Optional[torch.Tensor] = None
    ) -> tuple:
        feats = self.frontend(waveform)
        feat_mask: Optional[torch.Tensor] = None
        if wav_lengths is not None:
            frame_lengths = self.frontend.num_frames(wav_lengths)
            feat_mask = lengths_to_mask(frame_lengths, feats.shape[1]).to(feats.device)
        seq, mask = self.encoder(feats, feat_mask)
        return self.connector(seq, mask)

    def forward(
        self,
        waveform: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        wav_lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        prefix, prefix_mask = self.encode_audio(waveform, wav_lengths)
        token_embeds = self.lm.embed_tokens(input_ids)
        embeds = torch.cat([prefix, token_embeds], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, dtype=torch.bool, device=input_ids.device)
        full_mask = torch.cat([prefix_mask, attention_mask.bool()], dim=1)

        logits = self.lm(inputs_embeds=embeds, attention_mask=full_mask)
        n_prefix = prefix.shape[1]
        # logits at position (n_prefix - 1 + i) predict input_ids[:, i]
        text_logits = logits[:, n_prefix - 1 : -1, :]

        out: Dict[str, torch.Tensor] = {"logits": logits, "text_logits": text_logits}
        vocab = text_logits.shape[-1]
        loss = F.cross_entropy(
            text_logits.reshape(-1, vocab),
            input_ids.reshape(-1),
            ignore_index=self.tokenizer.pad_id,
        )
        out["loss"] = loss
        return out

    @torch.no_grad()
    def generate(
        self,
        waveform: torch.Tensor,
        decode_config: Optional[DecodeConfig] = None,
        wav_lengths: Optional[torch.Tensor] = None,
    ) -> List[str]:
        was_training = self.training
        self.eval()
        cfg = decode_config or DecodeConfig()
        prefix, prefix_mask = self.encode_audio(waveform, wav_lengths)
        bos, eos = self.tokenizer.bos_id, self.tokenizer.eos_id

        if cfg.strategy == "greedy":
            ids = greedy_decode(self.lm, prefix, prefix_mask, bos, eos, cfg.max_new_tokens)
        elif cfg.strategy == "beam":
            ids = beam_search(
                self.lm,
                prefix,
                prefix_mask,
                bos,
                eos,
                cfg.max_new_tokens,
                cfg.beam_size,
                cfg.length_penalty,
            )
        elif cfg.strategy == "sample":
            ids = sample_decode(
                self.lm,
                prefix,
                prefix_mask,
                bos,
                eos,
                cfg.max_new_tokens,
                cfg.temperature,
                cfg.top_k,
                cfg.top_p,
            )
        else:
            raise ValueError(f"unknown decode strategy '{cfg.strategy}'")

        self.train(was_training)
        return [self.tokenizer.decode(seq.tolist()) for seq in ids]

    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.forward(
            batch["waveform"],
            batch["input_ids"],
            batch.get("attention_mask"),
            batch.get("wav_lengths"),
        )["loss"]
