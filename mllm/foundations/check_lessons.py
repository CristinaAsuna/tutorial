"""Deterministic semantic checks, defaulting to the student's implementation."""
import argparse
import importlib
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from edu_core import attention, masks, training, vision


MODULES = {1: "lesson1_patchify", 2: "lesson2_tubelets", 3: "lesson3_attention",
           4: "lesson4_masks", 5: "lesson5_positions", 6: "lesson6_ema"}


def reference(number):
    if number in (1, 2, 5):
        return vision
    if number == 3:
        return SimpleNamespace(scaled_attention=lambda q, k, v, allowed=None:
                               F.scaled_dot_product_attention(q, k, v, attn_mask=allowed),
                               MultiHeadAttention=attention.MultiHeadAttention,
                               SelfAttentionBlock=attention.SelfAttentionBlock)
    if number == 4:
        return SimpleNamespace(causal_allow_mask=masks.causal_allow_mask,
                               select_tokens=lambda x, ids: x.gather(1, ids[..., None].expand(-1, -1, x.shape[-1])))
    return training


def check(number, impl):
    torch.manual_seed(7)
    if number == 1:
        x = torch.arange(48).reshape(1, 2, 4, 6)
        patches = impl.patchify(x, 2)
        assert torch.equal(patches[0, 0], torch.tensor([0, 24, 1, 25, 6, 30, 7, 31])), "局部次序应为 p,p,C"
        assert torch.equal(impl.unpatchify(patches, 2, 2, grid=(2, 3)), x), "矩形网格往返应完全一致"
    elif number == 2:
        video = torch.arange(32).reshape(1, 1, 4, 2, 4)
        patches = impl.tubelet_patchify(video, 2, 2)
        assert torch.equal(patches[0, 0], torch.tensor([0, 1, 4, 5, 8, 9, 12, 13])), "检查局部时间/空间轴"
        assert torch.equal(patches[0, 1], patches[0, 0] + 2), "宽度应变化最快"
        assert torch.equal(patches[0, 2], patches[0, 0] + 16), "时间网格应变化最慢"
    elif number == 3:
        q, k = torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2)
        v = torch.tensor([[[[2., 2.], [6., 6.]]]])
        allowed = torch.tensor([[[[True, False], [False, False]]]])
        assert torch.equal(impl.scaled_attention(q, k, v), torch.full_like(q, 4)), "softmax 应沿 key 轴"
        assert torch.equal(impl.scaled_attention(q, k, v, allowed), torch.tensor([[[[2., 2.], [0., 0.]]]])), "全遮蔽行应为零"
        oracle = attention.MultiHeadAttention(4, 2, kv_dim=6)
        student = impl.MultiHeadAttention(4, 2, kv_dim=6)
        student.load_state_dict(oracle.state_dict())
        query, context = torch.randn(2, 3, 4), torch.randn(2, 2, 6)
        mask = torch.tensor([[[True, False], [False, True], [False, False]]]).expand(2, -1, -1)
        assert torch.allclose(student(query, context, attention_mask=mask), oracle(query, context, attention_mask=mask), atol=1e-6), "检查跨维 cross-attention 与合头"
        block_oracle = attention.SelfAttentionBlock(4, 2)
        block = impl.SelfAttentionBlock(4, 2)
        block.load_state_dict(block_oracle.state_dict())
        assert torch.allclose(block(query), block_oracle(query), atol=1e-6), "检查 Pre-LN 和两条残差"
    elif number == 4:
        assert torch.equal(impl.causal_allow_mask(3), torch.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=torch.bool)), "不能看未来"
        x = torch.arange(12).reshape(2, 3, 2)
        ids = torch.tensor([[2, 0], [1, 2]])
        assert torch.equal(impl.select_tokens(x, ids), torch.tensor([[[4, 5], [0, 1]], [[8, 9], [10, 11]]])), "每个 batch 有自己的索引和顺序"
    elif number == 5:
        positions = torch.randn(1, 5, 4)
        result = impl.interpolate_2d_pos_embed(positions, (3, 2))
        assert torch.equal(result[:, :1], positions[:, :1]), "CLS 不应参与空间插值"
        assert torch.allclose(result, vision.interpolate_2d_pos_embed(positions, (3, 2)), atol=1e-6), "检查空间/通道轴"
    elif number == 6:
        teacher, student = nn.Linear(1, 1, bias=False), nn.Linear(1, 1, bias=False)
        for momentum, expected in [(0.75, 3.), (0., 6.), (1., 2.)]:
            with torch.no_grad():
                teacher.weight.fill_(2)
                student.weight.fill_(6)
            impl.update_ema(teacher, student, momentum)
            assert teacher.weight.item() == expected, "EMA 权重方向或原地更新不正确"
            assert teacher.weight.grad is None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lesson", type=int, choices=MODULES)
    parser.add_argument("--implementation", choices=("reference", "practice"), default="practice")
    args = parser.parse_args()
    numbers = [args.lesson] if args.lesson else list(MODULES)
    unfinished = False
    for number in numbers:
        impl = reference(number) if args.implementation == "reference" else importlib.import_module(MODULES[number])
        try:
            check(number, impl)
        except NotImplementedError as error:
            print(f"未完成：{MODULES[number]} — {error}。查看 HINTS.md，再重新检查。")
            unfinished = True
            continue
        print(f"基础 {number} ({args.implementation}) passed")
    return 2 if unfinished else 0


if __name__ == "__main__":
    raise SystemExit(main())
