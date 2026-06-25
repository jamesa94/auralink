import torch

from auralink.audio.augment import SpecAugment
from auralink.utils.seed import set_seed


def test_augment_is_noop_in_eval():
    aug = SpecAugment().eval()
    spec = torch.ones(1, 20, 16)
    assert torch.equal(aug(spec), spec)


def test_augment_masks_in_train():
    set_seed(0)
    aug = SpecAugment(n_time_masks=2, time_width=6, n_freq_masks=2, freq_width=4).train()
    spec = torch.ones(1, 40, 16)
    out = aug(spec)
    assert out.shape == spec.shape
    assert (out == 0.0).any()


def test_augment_preserves_shape_batch():
    aug = SpecAugment().train()
    spec = torch.randn(3, 25, 20)
    assert aug(spec).shape == spec.shape
