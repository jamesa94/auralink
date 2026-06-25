"""Command-line interface: ``auralink {info,caption,tag}``.

Without a checkpoint the ``caption`` / ``tag`` commands run a randomly
initialised model, which is enough to smoke-test the install end to end.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

import torch

from auralink._version import __version__
from auralink.audio.io import load_audio, synthetic_waveform
from auralink.bridge.registry import list_connectors
from auralink.config.presets import tiny_captioner_config, tiny_tagger_config
from auralink.data.ontology import SoundEventOntology
from auralink.decoding.base import DecodeConfig
from auralink.encoders.registry import list_encoders
from auralink.inference.pipeline import CaptionPipeline, TagPipeline
from auralink.llm.registry import list_language_models
from auralink.llm.tokenizer import CharTokenizer
from auralink.model.factory import build_captioner, build_tagger
from auralink.utils.checkpoint import load_state_into


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="auralink", description=__doc__)
    parser.add_argument("--version", action="version", version=f"auralink {__version__}")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("info", help="show version and registered components")

    caption = sub.add_parser("caption", help="caption an audio clip")
    caption.add_argument("audio", help="path to an audio file")
    caption.add_argument("--checkpoint", default=None, help="trained captioner checkpoint")
    caption.add_argument("--strategy", default="greedy", choices=["greedy", "beam", "sample"])
    caption.add_argument("--max-new-tokens", type=int, default=32)

    tag = sub.add_parser("tag", help="tag sound events in a clip")
    tag.add_argument("audio", help="path to an audio file")
    tag.add_argument("--checkpoint", default=None, help="trained tagger checkpoint")
    tag.add_argument("--top-k", type=int, default=5)
    return parser


def _load_or_synth(path: str, sample_rate: int) -> torch.Tensor:
    try:
        return load_audio(path, sample_rate)
    except Exception:
        print(f"could not load '{path}'; using a synthetic tone instead", file=sys.stderr)
        return synthetic_waveform(sample_rate, seed=0)


def _cmd_info() -> int:
    print(f"auralink {__version__}")
    print("encoders:", ", ".join(list_encoders()))
    print("connectors:", ", ".join(list_connectors()))
    print("language models:", ", ".join(list_language_models()))
    return 0


def _cmd_caption(args: argparse.Namespace) -> int:
    tokenizer = CharTokenizer.default()
    model = build_captioner(tiny_captioner_config(), tokenizer)
    if args.checkpoint:
        load_state_into(model, args.checkpoint)
    else:
        print("warning: no checkpoint; using a randomly initialised model", file=sys.stderr)
    wav = _load_or_synth(args.audio, 16000)
    pipeline = CaptionPipeline(
        model,
        decode_config=DecodeConfig(strategy=args.strategy, max_new_tokens=args.max_new_tokens),
    )
    print(pipeline(wav))
    return 0


def _cmd_tag(args: argparse.Namespace) -> int:
    ontology = SoundEventOntology.default()
    model = build_tagger(tiny_tagger_config(ontology.num_classes))
    if args.checkpoint:
        load_state_into(model, args.checkpoint)
    else:
        print("warning: no checkpoint; using a randomly initialised model", file=sys.stderr)
    wav = _load_or_synth(args.audio, 16000)
    pipeline = TagPipeline(model, ontology, threshold=0.0)
    for name, score in pipeline(wav, top_k=args.top_k):
        print(f"{score:.3f}\t{name}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "info":
        return _cmd_info()
    if args.command == "caption":
        return _cmd_caption(args)
    if args.command == "tag":
        return _cmd_tag(args)
    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
