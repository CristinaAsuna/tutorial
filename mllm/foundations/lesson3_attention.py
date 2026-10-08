"""基础 3：attention、多头与 Pre-LN 残差。

前置：矩阵乘法、softmax；dim=-1 是 key 轴。
公式：softmax(QKᵀ/√d + mask)V；True 表示允许看见。
形状：Q=(B,H,Q,d)，K/V=(B,H,K,d)，分数=(B,H,Q,K)。
例子：Q/K 全零、V=[2,6]，允许两个 key 时结果 4，只允许第一个时结果 2。
常见错：softmax 在 query 轴；全遮蔽行变成均匀分布；合头遗漏 transpose。
接入：论文直接复用 edu_core.MultiHeadAttention/SelfAttentionBlock。
检查：python3 check_lessons.py --lesson 3；提示：HINTS.md 的基础 3。
"""
import math
import torch
from edu_core.attention import MultiHeadAttention as ReferenceAttention, SelfAttentionBlock as ReferenceBlock


def scaled_attention(q, k, v, allowed=None):
    # TODO 3.1：Q @ K.transpose(-2,-1) / sqrt(d)。
    # TODO 3.2：allowed=False 的分数填为 dtype 最小值，再对 key 轴 softmax。
    # TODO 3.3：全遮蔽 query 的权重置零，返回 weights @ V。
    raise NotImplementedError("基础 3.1–3.3：scaled_attention")


class MultiHeadAttention(ReferenceAttention):
    """继承的 __init__ 已提供 q/k/v/out_proj；这里只填写 forward。"""
    def forward(self, query, context=None, *, attention_mask=None):
        # TODO 3.4：context=None 时取 query，分别投影 Q/K/V。
        # TODO 3.5：拆为 (B,H,S,d)；2D/3D mask 补成可广播的 4D。
        # TODO 3.6：调用本文件 scaled_attention，转置合头并 out_proj。
        # TODO 3.7：全遮蔽行的输出也置零，消除 out_proj.bias。
        raise NotImplementedError("基础 3.4–3.7：MultiHeadAttention.forward")


class SelfAttentionBlock(ReferenceBlock):
    def __init__(self, dim=4, num_heads=2):
        super().__init__(dim, num_heads)
        self.self_attn = MultiHeadAttention(dim, num_heads)

    def forward(self, x):
        # TODO 3.8：x = x + self_attn(norm1(x))，保留原始残差。
        # TODO 3.9：x = x + mlp(norm2(x))。
        raise NotImplementedError("基础 3.8–3.9：SelfAttentionBlock.forward")
