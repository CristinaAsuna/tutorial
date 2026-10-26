"""Offline contracts and independent attention checks. No pretrained model is downloaded."""
import subprocess
import sys
import torch
import pytest
from edu_core.backbones import TinyBackbone, DecoderSpec, prefix_encoding, require_capabilities
from edu_core.backbones.layers import projected_attention


def test_optional_dependencies_are_lazy():
    subprocess.run([sys.executable, "-c", "import sys; from edu_core.backbones import TinyBackbone; assert 'transformers' not in sys.modules"], check=True)


def test_spec_and_capability_rejection():
    with pytest.raises(ValueError):
        DecoderSpec(16, 2, 3, 2, 4, 32)
    with pytest.raises(ValueError, match="nonexistent"):
        require_capabilities(TinyBackbone(), "nonexistent")


def test_gqa_hand_value_and_empty_row():
    q = torch.zeros(1, 2, 2, 2)
    k = torch.zeros(1, 1, 2, 2)
    v = torch.tensor([[[[2., 4.], [6., 8.]]]])
    allow = torch.tensor([[[True, True], [False, False]]])
    got = projected_attention(q, k, v, allow)
    torch.testing.assert_close(got, torch.tensor([[[4., 6., 4., 6.], [0., 0., 0., 0.]]]))


def test_vision_raw_projected_and_gradients():
    model = TinyBackbone()
    images = torch.randn(2, 3, 8, 8, requires_grad=True)
    raw = model.encode_vision(images)
    projected = model.encode_vision(images, projected=True)
    assert raw.grid == (2, 2) and raw.cls_index is None
    assert raw.tokens.shape == (2, 4, 24) and projected.tokens.shape == (2, 4, 64)
    projected.tokens.sum().backward()
    assert images.grad.abs().sum() > 0


def test_causal_vs_bidirectional_prefix_and_padding():
    model = TinyBackbone().eval()
    x = torch.randn(1, 4, 64)
    valid = torch.tensor([[True, True, True, False]])
    original = model.language(x, valid).hidden_states
    changed = x.clone(); changed[:, 2] += torch.arange(64)
    torch.testing.assert_close(model.language(changed, valid).hidden_states[:, 0], original[:, 0])
    allow = valid[:, :, None] & valid[:, None, :]
    bidir = model.language(x, valid, allow_mask=allow).hidden_states
    assert not torch.allclose(model.language(changed, valid, allow_mask=allow).hidden_states[:, 0], bidir[:, 0])
    changed = x.clone(); changed[:, 3] += 50
    torch.testing.assert_close(model.language(changed, valid, allow_mask=allow).hidden_states[:, :3], bidir[:, :3])


def test_prefix_kv_keeps_gradient_and_new_observation_is_fresh():
    model = TinyBackbone()
    x = torch.randn(2, 4, 64, requires_grad=True)
    valid = torch.ones(2, 4, dtype=torch.bool)
    first = prefix_encoding(model, x, valid)
    assert len(first.key_values) == 2
    assert first.key_values[0][0].shape == (2, 2, 4, 16)
    sum(k.square().mean()+v.square().mean() for k, v in first.key_values).backward()
    assert x.grad.abs().sum() > 0
    second = prefix_encoding(model, x.detach()+1, valid)
    assert first.key_values[0][0].data_ptr() != second.key_values[0][0].data_ptr()
    assert not torch.equal(first.key_values[0][0], second.key_values[0][0])
