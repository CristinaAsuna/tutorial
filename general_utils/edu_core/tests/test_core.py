import torch
from torch import nn

from edu_core.attention import MultiHeadAttention
from edu_core.batching import IGNORE_INDEX, expand_single_image_token, right_pad
from edu_core.masks import causal_allow_mask
from edu_core.training import freeze_and_keep_eval, update_ema
from edu_core.vision import PatchEmbed, interpolate_2d_pos_embed


def test_causal_mask_disallows_future_keys():
    mask = causal_allow_mask(3)
    assert torch.equal(mask, torch.tensor([[True, False, False], [True, True, False], [True, True, True]]))


def test_attention_ignores_masked_key_values():
    torch.manual_seed(0)
    attn = MultiHeadAttention(4, 2).eval()
    x = torch.randn(1, 3, 4)
    changed = x.clone()
    changed[:, 2] = 10_000
    valid = torch.tensor([[1, 1, 0]])
    assert torch.allclose(attn(x, key_padding_mask=valid)[:, :2], attn(changed, key_padding_mask=valid)[:, :2], atol=1e-5)


def test_patch_embed_and_position_interpolation():
    tokens, grid = PatchEmbed(3, 8, 4)(torch.randn(2, 3, 8, 8))
    assert tokens.shape == (2, 4, 8) and grid == (2, 2)
    assert interpolate_2d_pos_embed(torch.randn(1, 5, 8), (3, 2)).shape == (1, 7, 8)


def test_right_padding_and_image_expansion_mask_labels():
    ids, valid = right_pad([torch.tensor([1, -200, 2]), torch.tensor([1, -200])], pad_value=0)
    labels = torch.tensor([[-100, -100, 2], [-100, -100, -100]])
    embedding = nn.Embedding(10, 4)
    images = torch.randn(2, 2, 4)
    packed, mask, out_labels = expand_single_image_token(ids, images, embedding, image_token_index=-200, attention_mask=valid, labels=labels)
    assert packed.shape == (2, 4, 4)
    assert torch.all(out_labels[:, 1:3] == IGNORE_INDEX)
    assert torch.equal(mask[1], torch.tensor([True, True, True, False]))


def test_freeze_and_ema():
    student, teacher = nn.Linear(2, 2), nn.Linear(2, 2)
    teacher.load_state_dict(student.state_dict())
    freeze_and_keep_eval(teacher)
    assert not teacher.training and all(not p.requires_grad for p in teacher.parameters())
    before = teacher.weight.detach().clone()
    with torch.no_grad():
        student.weight.add_(2)
    update_ema(teacher, student, 0.5)
    assert torch.allclose(teacher.weight, before + 1)
