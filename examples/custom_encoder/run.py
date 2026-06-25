"""Register a custom audio encoder and drop it into the captioner via config."""

import torch
import torch.nn as nn

from auralink import CharTokenizer, build_captioner, tiny_captioner_config
from auralink.encoders.base import AudioEncoder
from auralink.encoders.registry import list_encoders, register_encoder


@register_encoder("bilstm")
class BiLSTMEncoder(AudioEncoder):
    """A bidirectional LSTM encoder, registered under the name ``bilstm``.

    Extra keyword arguments coming from ``EncoderConfig`` (n_heads, ff_dim, ...)
    are accepted and ignored so the factory can build it like any other encoder.
    """

    def __init__(self, input_dim=40, d_model=128, n_layers=1, **kwargs):
        super().__init__()
        self._output_dim = d_model
        self.proj = nn.Linear(input_dim, d_model)
        self.lstm = nn.LSTM(
            d_model, d_model // 2, num_layers=n_layers, batch_first=True, bidirectional=True
        )

    @property
    def output_dim(self):
        return self._output_dim

    def forward(self, features, mask=None):
        hidden = self.proj(features)
        hidden, _ = self.lstm(hidden)
        return hidden, mask


def main() -> None:
    print("registered encoders:", list_encoders())

    tokenizer = CharTokenizer.default()
    config = tiny_captioner_config()
    config.encoder.name = "bilstm"
    model = build_captioner(config, tokenizer)

    waveform = torch.randn(1, 16000)
    input_ids = torch.tensor([tokenizer.encode("a test clip")])
    loss = model(waveform, input_ids)["loss"]
    print("forward loss with custom encoder:", float(loss.detach()))


if __name__ == "__main__":
    main()
