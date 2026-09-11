"""
lesson1_attention_and_block.py
==============================
【关卡 1】：Q-Former 基础积木 (Multi-Head Attention 与 QFormerLayer)

本文件包含 3 个组件，请你依次实现它们的 TODO:
1. MultiHeadAttention: 支持 Self-Attention 与 Cross-Attention
2. FeedForward: 两层 MLP 结构
3. QFormerLayer: 【核心机制】仅允许前 M 个 Query 查图像，Text 不查图像！

测试命令:
  /Users/max/codebase/.ml/.venv/bin/python lesson1_attention_and_block.py
"""

import math
import torch
import torch.nn as nn
from typing import Optional


# ==============================================================================
# 组件 1: 通用 Multi-Head Attention
# ==============================================================================
class MultiHeadAttention(nn.Module):
    """
    【设计意图】:
      既能用于 Self-Attention (Query 和 Text 内部交互)，
      又能用于 Cross-Attention (Query 查图像，此时图像的 kv_dim 可能与 hidden_size 不同)。

    【__init__ 参数】:
      - hidden_size: Query 的隐层维度 (如 768)
      - num_heads: 多头数量 (如 12)
      - kv_dim: Key/Value 的输入维度 (如图像特征是 1408 维，若为 None 则等于 hidden_size)

    【self. 需定义的属性】:
      - self.head_dim = hidden_size // num_heads (每个头的维度)
      - self.scale = 1.0 / math.sqrt(self.head_dim) (缩放系数)
      - self.q_proj: Linear(hidden_size, hidden_size)
      - self.k_proj: Linear(kv_dim, hidden_size)
      - self.v_proj: Linear(kv_dim, hidden_size)
      - self.out_proj: Linear(hidden_size, hidden_size)
    """
    def __init__(self, hidden_size: int, num_heads: int, kv_dim: Optional[int] = None):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size 必须能被 num_heads 整除"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        kv_dim = kv_dim if kv_dim is not None else hidden_size

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
        【输入 Tensor 维度】:
          - hidden_states: (B, S_q, D_q) 作为 Query
          - context: (B, S_kv, D_kv) 作为 Key/Value。若为 None 则等于 hidden_states (Self-Attention)
          - attention_mask: (B, 1, S_q, S_kv) 或 (B, S_q, S_kv)，1 为保留，0 为遮蔽

        【输出 Tensor 维度】:
          - out: (B, S_q, D_q)
        """
        # =========================================================================
        # TODO 1.1: 请实现 Multi-Head Attention 前向计算
        # 步骤提示:
        #   1. 若 context is None 则 context = hidden_states
        #   2. 获取 B, S_q = hidden_states.shape[:2], S_kv = context.shape[1]
        #   3. Q, K, V 分别投影并变换维度为: (B, num_heads, Seq_len, head_dim)
        #   4. attn_scores = (Q @ K^T) * self.scale -> 形状: (B, num_heads, S_q, S_kv)
        #   5. 如果 attention_mask is not None:
        #        使用 attn_scores.masked_fill(attention_mask == 0, -1e9)
        #   6. attn_weights = softmax(attn_scores, dim=-1)
        #   7. out = attn_weights @ V -> 形状: (B, num_heads, S_q, head_dim)
        #   8. 转置并合并多头为 (B, S_q, hidden_size)，过 self.out_proj 返回
        # =========================================================================
        raise NotImplementedError("TODO 1.1 尚未实现！请实现 MultiHeadAttention.forward")


# ==============================================================================
# 组件 2: 前馈神经网络 (FFN)
# ==============================================================================
class FeedForward(nn.Module):
    """
    【设计意图】: 标准两层 Transformer MLP (升维 -> 激活 -> 降维)
    【__init__ 参数】:
      - hidden_size: 输入与输出维度 (如 768)
      - intermediate_size: 中间隐藏层升维大小 (如 3072)
      - dropout: 随机丢弃率
    """
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        【输入】: x: (B, S, D_q)
        【输出】: (B, S, D_q)
        """
        # =========================================================================
        # TODO 1.2: 请实现两层 MLP 前向计算
        # 计算流程: x -> fc1 -> act -> fc2 -> dropout -> 返回
        # =========================================================================
        raise NotImplementedError("TODO 1.2 尚未实现！请实现 FeedForward.forward")


# ==============================================================================
# 组件 3: QFormerLayer (Q-Former 单层完整结构)
# ==============================================================================
class QFormerLayer(nn.Module):
    """
    【核心机制】:
      - Self-Attention: 全体 Token (Query + Text 共 M+L 个) 共同参与，受 Mask 控制
      - Cross-Attention: 【绝密细节】只有前 M 个 Query 可以查图像，Text 保持原样！
      - FFN: 全体 Token 经过前馈网络
    """
    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        intermediate_size: int = 3072,
        img_feat_dim: int = 1408,
        has_cross_attention: bool = True
    ):
        super().__init__()
        self.has_cross_attention = has_cross_attention

        # 1. 共享的 Self-Attention
        self.self_attn = MultiHeadAttention(hidden_size, num_heads)
        self.norm1 = nn.LayerNorm(hidden_size)

        # 2. 针对图像的 Cross-Attention
        if self.has_cross_attention:
            self.cross_attn = MultiHeadAttention(hidden_size, num_heads, kv_dim=img_feat_dim)
            self.norm2 = nn.LayerNorm(hidden_size)

        # 3. 前馈层
        self.ffn = FeedForward(hidden_size, intermediate_size)
        self.norm3 = nn.LayerNorm(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_query_tokens: int,
        attention_mask: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        【输入 Tensor 维度】:
          - hidden_states: (B, M + L, D_q) [前 M 个是 Query, 后 L 个是 Text]
          - num_query_tokens: 整数 M (如 32)
          - attention_mask: (B, 1, M+L, M+L) 自定义可见性掩码
          - image_embeds: (B, N_img, D_img) 冻结 ViT 图像特征

        【输出 Tensor 维度】:
          - (B, M + L, D_q)
        """
        # =========================================================================
        # TODO 1.3: 请实现 QFormerLayer 的前向传播
        # 步骤 1 (Self-Attention):
        #   残差连接 + LayerNorm:
        #   hidden_states = hidden_states + self.self_attn(self.norm1(hidden_states), attention_mask=attention_mask)
        #
        # 步骤 2 (Cross-Attention - 仅对 Query 生效!):
        #   如果 self.has_cross_attention 且 image_embeds is not None:
        #       a. 切分出 Query: query_tokens = hidden_states[:, :num_query_tokens, :]
        #       b. 对 query_tokens 做 LayerNorm 与 Cross-Attention:
        #          query_out = query_tokens + self.cross_attn(self.norm2(query_tokens), context=image_embeds)
        #       c. 拼接回完整序列:
        #          如果有文本 (hidden_states 长度 > num_query_tokens):
        #              hidden_states = torch.cat([query_out, hidden_states[:, num_query_tokens:, :]], dim=1)
        #          否则:
        #              hidden_states = query_out
        #
        # 步骤 3 (FFN):
        #   残差连接 + LayerNorm:
        #   hidden_states = hidden_states + self.ffn(self.norm3(hidden_states))
        # =========================================================================
        raise NotImplementedError("TODO 1.3 尚未实现！请实现 QFormerLayer.forward")


# ==============================================================================
# 单元测试验证
# ==============================================================================
def run_test():
    print("\n========== 开始测试关卡 1 ==========")
    B, M, L, D_q = 2, 4, 3, 768
    num_heads = 12
    N_img, D_img = 16, 1408

    # 1. 测试 MHA
    mha = MultiHeadAttention(D_q, num_heads)
    x = torch.randn(B, M + L, D_q)
    out_mha = mha(x)
    assert out_mha.shape == (B, M + L, D_q), f"MHA 形状不符: {out_mha.shape}"
    print("✅ TODO 1.1 (MultiHeadAttention) 通过测试！")

    # 2. 测试 FFN
    ffn = FeedForward(D_q, 3072)
    out_ffn = ffn(x)
    assert out_ffn.shape == (B, M + L, D_q), f"FFN 形状不符: {out_ffn.shape}"
    print("✅ TODO 1.2 (FeedForward) 通过测试！")

    # 3. 测试 QFormerLayer
    layer = QFormerLayer(hidden_size=D_q, num_heads=num_heads, img_feat_dim=D_img)
    img_feat = torch.randn(B, N_img, D_img)
    mask = torch.ones(B, 1, M + L, M + L)
    out_layer = layer(x, num_query_tokens=M, attention_mask=mask, image_embeds=img_feat)
    assert out_layer.shape == (B, M + L, D_q), f"QFormerLayer 形状不符: {out_layer.shape}"
    print("✅ TODO 1.3 (QFormerLayer) 通过测试！")
    print("🎉 恭喜！关卡 1 全部挑战成功！\n")


if __name__ == "__main__":
    run_test()
