"""Assemble captioners and taggers from their dataclass configs."""

from __future__ import annotations

from auralink.audio.features import LogMelFrontend
from auralink.bridge.base import Connector
from auralink.bridge.registry import build_connector
from auralink.encoders.base import AudioEncoder
from auralink.encoders.registry import build_encoder
from auralink.llm.base import LanguageModel
from auralink.llm.registry import build_language_model
from auralink.llm.tokenizer import CharTokenizer
from auralink.model.captioner import AudioCaptioner
from auralink.model.config import (
    CaptionerConfig,
    ConnectorConfig,
    EncoderConfig,
    FrontendConfig,
    LMConfig,
    TaggerConfig,
)
from auralink.model.tagger import SoundEventTagger


def _build_frontend(cfg: FrontendConfig) -> LogMelFrontend:
    return LogMelFrontend(
        sample_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        f_min=cfg.f_min,
        f_max=cfg.f_max,
    )


def _build_encoder(cfg: EncoderConfig, input_dim: int) -> AudioEncoder:
    return build_encoder(
        cfg.name,
        input_dim=input_dim,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        ff_dim=cfg.ff_dim,
        conv_stride=cfg.conv_stride,
        n_conv=cfg.n_conv,
        dropout=cfg.dropout,
    )


def _build_connector(cfg: ConnectorConfig, input_dim: int, output_dim: int) -> Connector:
    kwargs = {"input_dim": input_dim, "output_dim": output_dim, "dropout": cfg.dropout}
    if cfg.name == "linear":
        kwargs["stack_size"] = cfg.stack_size
    elif cfg.name == "pooling":
        kwargs.update(num_query_tokens=cfg.num_query_tokens, n_heads=cfg.n_heads)
    elif cfg.name == "qformer":
        kwargs.update(
            num_query_tokens=cfg.num_query_tokens, n_heads=cfg.n_heads, n_layers=cfg.n_layers
        )
    return build_connector(cfg.name, **kwargs)


def _build_lm(cfg: LMConfig, vocab_size: int) -> LanguageModel:
    if cfg.name == "tiny":
        return build_language_model(
            "tiny",
            vocab_size=vocab_size,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            ff_dim=cfg.ff_dim,
            max_len=cfg.max_len,
            dropout=cfg.dropout,
        )
    if cfg.name == "hf":
        if cfg.model_name_or_path is None:
            raise ValueError("LMConfig.model_name_or_path is required for the 'hf' language model")
        return build_language_model("hf", model_name_or_path=cfg.model_name_or_path)
    raise ValueError(f"unknown language model '{cfg.name}'")


def build_captioner(config: CaptionerConfig, tokenizer: CharTokenizer) -> AudioCaptioner:
    """Construct an :class:`AudioCaptioner` from a config and a tokenizer.

    The connector is always built to project into the LM hidden size so the
    audio prefix can be concatenated with the text embeddings.
    """
    frontend = _build_frontend(config.frontend)
    encoder = _build_encoder(config.encoder, input_dim=config.frontend.n_mels)
    lm = _build_lm(config.lm, vocab_size=tokenizer.vocab_size)
    connector = _build_connector(
        config.connector, input_dim=encoder.output_dim, output_dim=lm.hidden_size
    )
    return AudioCaptioner(
        frontend, encoder, connector, lm, tokenizer, max_caption_len=config.max_caption_len
    )


def build_tagger(config: TaggerConfig) -> SoundEventTagger:
    """Construct a :class:`SoundEventTagger` from a config."""
    frontend = _build_frontend(config.frontend)
    encoder = _build_encoder(config.encoder, input_dim=config.frontend.n_mels)
    return SoundEventTagger(
        frontend,
        encoder,
        num_classes=config.num_classes,
        pool=config.pool,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    )
