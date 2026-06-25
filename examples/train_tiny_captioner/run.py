"""Overfit the tiny captioner on a few synthetic clips to show training works."""

import torch
from torch.utils.data import DataLoader

from auralink import CharTokenizer, DecodeConfig, build_captioner, tiny_captioner_config
from auralink.data import CaptionCollator, CaptionDataset, ManifestItem
from auralink.training import Trainer
from auralink.utils.seed import set_seed


def main() -> None:
    set_seed(0)
    tokenizer = CharTokenizer.default()
    model = build_captioner(tiny_captioner_config(), tokenizer)

    captions = ["a dog is barking", "rain on the roof", "a car drives past"]
    items = [
        ManifestItem(audio_path=f"clip{i}.wav", captions=[caption], uid=str(i))
        for i, caption in enumerate(captions)
    ]
    dataset = CaptionDataset(items, tokenizer, sample_rate=16000, max_caption_len=32)
    loader = DataLoader(dataset, batch_size=3, collate_fn=CaptionCollator(tokenizer.pad_id))

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    trainer = Trainer(model, optimizer)
    for epoch in range(30):
        loss = trainer.train_epoch(loader)["loss"]
        if epoch % 10 == 0:
            print(f"epoch {epoch:2d}  loss {loss:.3f}")

    waveform = dataset[0]["waveform"].unsqueeze(0)
    caption = model.generate(waveform, DecodeConfig(max_new_tokens=24))[0]
    print("sample caption after overfitting:", caption)


if __name__ == "__main__":
    main()
