import torch
from torch.utils.data import DataLoader

from auralink.data.caption_dataset import CaptionDataset
from auralink.data.collate import CaptionCollator, collate_tags
from auralink.data.manifest import ManifestItem
from auralink.data.ontology import SoundEventOntology
from auralink.data.tagging_dataset import TaggingDataset
from auralink.llm.tokenizer import CharTokenizer
from auralink.model.config import (
    CaptionerConfig,
    ConnectorConfig,
    EncoderConfig,
    FrontendConfig,
    LMConfig,
    TaggerConfig,
)
from auralink.model.factory import build_captioner, build_tagger
from auralink.training.schedule import get_warmup_cosine_scheduler
from auralink.training.trainer import Trainer
from auralink.utils.seed import set_seed


def _caption_cfg():
    return CaptionerConfig(
        frontend=FrontendConfig(n_mels=32, n_fft=256, hop_length=128),
        encoder=EncoderConfig(d_model=48, n_layers=1, ff_dim=64),
        connector=ConnectorConfig(name="linear", stack_size=2),
        lm=LMConfig(d_model=48, n_layers=1, ff_dim=64, max_len=128),
        max_caption_len=16,
    )


def test_captioner_loss_decreases_on_fixed_batch():
    set_seed(0)
    tok = CharTokenizer.default()
    model = build_captioner(_caption_cfg(), tok)
    items = [
        ManifestItem(audio_path=f"m{i}.wav", captions=["a dog barks"], uid=str(i)) for i in range(4)
    ]
    ds = CaptionDataset(items, tok, sample_rate=4000, max_caption_len=16)
    loader = DataLoader(ds, batch_size=4, collate_fn=CaptionCollator(tok.pad_id))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_steps=2, total_steps=20)
    trainer = Trainer(model, optimizer, scheduler=scheduler, grad_clip=1.0)
    first = trainer.train_epoch(loader)["loss"]
    last = first
    for _ in range(10):
        last = trainer.train_epoch(loader)["loss"]
    assert last < first


def test_tagger_trains_and_eval_runs():
    set_seed(0)
    onto = SoundEventOntology.default()
    cfg = TaggerConfig(
        frontend=FrontendConfig(n_mels=32, n_fft=256, hop_length=128),
        encoder=EncoderConfig(d_model=48, n_layers=1, ff_dim=64),
        num_classes=onto.num_classes,
        hidden_dim=32,
    )
    model = build_tagger(cfg)
    items = [ManifestItem(audio_path=f"t{i}.wav", labels=["dog"], uid=str(i)) for i in range(4)]
    ds = TaggingDataset(items, onto, sample_rate=4000)
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_tags)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(model, optimizer)
    history = trainer.fit(loader, epochs=3, val_dataloader=loader)
    assert len(history) == 3
    assert "val_loss" in history[-1]
