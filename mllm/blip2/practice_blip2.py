"""
practice_blip2.py
=================
你的 BLIP-2 手写实战练手场！
你可以按照引导逐步实现各个组件，随时运行本文件进行单测断言。

运行方式:
  /Users/max/codebase/.ml/.venv/bin/python practice_blip2.py
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class MultiHeadAttention(nn.Module):
    """
    第一关：通用 Multi-Head Attention (支持 Self-Attention 和 Cross-Attention)
    """
    def __init__(self, hidden_size: int, num_heads: int, kv_dim: Optional[int] = None):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size 必须能被 num_heads 整除"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        kv_dim = kv_dim if kv_dim is not None else hidden_size

        # 1. 定义 Q, K, V 投影层以及输出投影层
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(kv_dim, hidden_size)
        self.v_proj = nn.Linear(kv_dim, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        请实现以下张量变化:
          hidden_states: (B, S_q, D_q)
          context:       (B, S_kv, D_kv) 如果为 None 则等于 hidden_states
          attention_mask:(B, 1, S_q, S_kv) 或 (B, S_q, S_kv) 可选

          返回: (B, S_q, D_q)
        """
        # =========================================================================
        # TODO: 请实现多头注意力的前向计算
        # 步骤提示:
        # 1. context 处理与获取 B, S_q, S_kv
        # 2. Q, K, V 投影并分头变形为 (B, num_heads, Seq_len, head_dim)
        # 3. 计算 attn_scores = (Q @ K.transpose(-1, -2)) * self.scale
        # 4. 若 attention_mask 不为空，使用 masked_fill 把 mask == 0 的地方设为 -1e9
        # 5. 计算 attn_weights = softmax(attn_scores, dim=-1)，并与 V 相乘
        # 6. 多头拼接，恢复为 (B, S_q, hidden_size)，经过 self.out_proj 并返回
        # =========================================================================
        raise NotImplementedError("请实现 MultiHeadAttention.forward 逻辑！")


def test_mha():
    print("\n--- 正在测试 MultiHeadAttention ---")
    B, S_q, D_q = 2, 8, 768
    num_heads = 12

    mha = MultiHeadAttention(hidden_size=D_q, num_heads=num_heads)
    x = torch.randn(B, S_q, D_q)

    # 1. 测试 Self-Attention
    out = mha(x)
    assert out.shape == (B, S_q, D_q), f"Self-Attention 输出形状错误: 期望 {(B, S_q, D_q)}, 实际得到 {out.shape}"
    print("✅ 1. Self-Attention 维度通过:", out.shape)

    # 2. 测试带 Mask 的 Self-Attention
    mask = torch.ones(B, 1, S_q, S_q)
    mask[:, :, :, -2:] = 0  # 遮蔽后 2 个 token
    out_masked = mha(x, attention_mask=mask)
    assert out_masked.shape == (B, S_q, D_q), "带 Mask 的 Self-Attention 形状错误"
    print("✅ 2. 带 Mask 的 Self-Attention 通过")

    # 3. 测试 Cross-Attention (不同 kv_dim)
    S_kv, D_kv = 16, 1408
    cross_mha = MultiHeadAttention(hidden_size=D_q, num_heads=num_heads, kv_dim=D_kv)
    context = torch.randn(B, S_kv, D_kv)
    cross_out = cross_mha(x, context=context)
    assert cross_out.shape == (B, S_q, D_q), f"Cross-Attention 输出形状错误: 期望 {(B, S_q, D_q)}, 实际得到 {cross_out.shape}"
    print("✅ 3. Cross-Attention (KV 维度不同) 通过:", cross_out.shape)
    print("🎉 恭喜！MultiHeadAttention 全部通过！")


if __name__ == "__main__":
    test_mha()
