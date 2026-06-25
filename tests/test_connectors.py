import pytest
import torch

from auralink.bridge import build_connector, list_connectors
from auralink.bridge.linear import LinearConnector
from auralink.bridge.qformer import QFormerConnector
from auralink.utils.masking import lengths_to_mask


def test_registry_has_all_connectors():
    for name in ("linear", "pooling", "qformer"):
        assert name in list_connectors()


def test_linear_connector_shapes():
    conn = LinearConnector(input_dim=64, output_dim=32, stack_size=2)
    prefix, mask = conn(torch.randn(2, 9, 64))
    assert prefix.shape[0] == 2
    assert prefix.shape[2] == 32
    assert mask.shape == (2, prefix.shape[1])
    assert prefix.shape[1] == 5  # ceil(9 / 2)


@pytest.mark.parametrize("name", ["pooling", "qformer"])
def test_fixed_token_connectors(name):
    conn = build_connector(name, input_dim=48, output_dim=32, num_query_tokens=8)
    prefix, mask = conn(torch.randn(2, 13, 48))
    assert prefix.shape == (2, 8, 32)
    assert mask.shape == (2, 8)
    assert bool(mask.all())


def test_qformer_with_padding_mask():
    conn = QFormerConnector(input_dim=40, output_dim=24, num_query_tokens=6, n_layers=2)
    mask = lengths_to_mask(torch.tensor([20, 8]), 20)
    prefix, _ = conn(torch.randn(2, 20, 40), mask)
    assert prefix.shape == (2, 6, 24)
    assert torch.isfinite(prefix).all()
