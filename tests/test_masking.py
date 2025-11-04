import torch

from auralink.utils.masking import (
    conv_output_length,
    downsample_lengths,
    lengths_to_mask,
    pad_to_multiple,
)


def test_lengths_to_mask():
    mask = lengths_to_mask(torch.tensor([2, 4]), max_len=4)
    expected = torch.tensor([[True, True, False, False], [True, True, True, True]])
    assert torch.equal(mask, expected)


def test_lengths_to_mask_infers_max():
    mask = lengths_to_mask(torch.tensor([1, 3]))
    assert mask.shape == (2, 3)


def test_downsample_lengths_uses_ceiling():
    out = downsample_lengths(torch.tensor([10, 7, 8]), factor=4)
    assert out.tolist() == [3, 2, 2]


def test_conv_output_length_matches_pytorch():
    conv = torch.nn.Conv1d(1, 1, kernel_size=3, stride=2, padding=1)
    out = conv(torch.zeros(1, 1, 17))
    assert out.shape[-1] == conv_output_length(17, 3, 2, 1)


def test_pad_to_multiple():
    x = torch.zeros(2, 5, 3)
    padded = pad_to_multiple(x, multiple=4, dim=1)
    assert padded.shape == (2, 8, 3)
