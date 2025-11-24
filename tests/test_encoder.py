import torch

from auralink.encoders import build_encoder, list_encoders
from auralink.encoders.conv_transformer import ConvTransformerEncoder
from auralink.encoders.pooling import AttentivePool, MeanPool
from auralink.utils.masking import lengths_to_mask


def test_registry_knows_conv_transformer():
    assert "conv_transformer" in list_encoders()


def test_encoder_forward_shape_and_downsample():
    enc = ConvTransformerEncoder(input_dim=64, d_model=96, n_layers=2)
    feats = torch.randn(2, 48, 64)
    out, mask = enc(feats)
    assert out.shape[0] == 2
    assert out.shape[2] == 96
    assert out.shape[1] < 48  # conv stem downsamples
    assert mask is None


def test_encoder_updates_mask():
    enc = build_encoder("conv_transformer", input_dim=32, d_model=64)
    feats = torch.randn(2, 40, 32)
    mask = lengths_to_mask(torch.tensor([40, 16]), 40)
    out, new_mask = enc(feats, mask)
    assert new_mask is not None
    assert new_mask.shape == (2, out.shape[1])
    assert new_mask[1].sum() < new_mask[0].sum()


def test_pooling_layers():
    x = torch.randn(2, 10, 8)
    mask = lengths_to_mask(torch.tensor([10, 5]), 10)
    assert MeanPool()(x, mask).shape == (2, 8)
    assert AttentivePool(8)(x, mask).shape == (2, 8)
