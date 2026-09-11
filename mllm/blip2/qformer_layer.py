"""
qformer_layer.py
================
Q-Former 的单个 Transformer 块 (QFormerLayer)

【核心架构特性】:
1. Self-Attention: Query Tokens 与 Text Tokens 拼成一个序列共同参与，受到三种自定义 Mask 的控制。
2. Cross-Attention: 【最为关键】只有前 M 个 Query Tokens 会和视觉编码器的输出做 Cross-Attention！
   后 L 个 Text Tokens 不直接接触图像，必须通过 Query 间接获取图像信息（这就是 Q-Former 的信息瓶颈设计）。
3. Feed-Forward Network (FFN): 标准的两层升维/降维 MLP。

数据流与 Tensor 形状变迁:
  输入 hidden_states: (B, M + L, D_q)
  输入 image_embeds:  (B, N_img, D_img)
  输出 hidden_states: (B, M + L, D_q)
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class MultiHeadAttention(nn.Module):
    """
    自研的通用 Multi-Head Attention，支持 Self-Attention 和 Cross-Attention。
    """
    def __init__(self, hidden_size: int, num_heads: int, kv_dim: Optional[int] = None):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size 必须能被 num_heads 整除"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # 如果是 Cross-Attention，KV 的维度可能与 Query 维度不同（如图像特征维度 1408 != 768）
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
        Tensor 形状演变:
        --------------------------------------------------------------------------
        如果作为 Self-Attention:
          hidden_states: (B, S_q, D_q) 其中 S_q = M + L
          context: None (默认就是 hidden_states)
        如果作为 Cross-Attention:
          hidden_states: (B, M, D_q) 只有查询向量作为 Query
          context: (B, N_img, D_img) 图像特征作为 Key 和 Value
        --------------------------------------------------------------------------
        """
        B, S_q, _ = hidden_states.shape
        if context is None:
            context = hidden_states
        B, S_kv, _ = context.shape

        # 1. 线性投影与拆分多头
        # Q: (B, S_q, D_q) -> (B, S_q, num_heads, head_dim) -> (B, num_heads, S_q, head_dim)
        q = self.q_proj(hidden_states).view(B, S_q, self.num_heads, self.head_dim).transpose(1, 2)

        # K, V: (B, S_kv, D_kv) -> (B, S_kv, num_heads, head_dim) -> (B, num_heads, S_kv, head_dim)
        k = self.k_proj(context).view(B, S_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(B, S_kv, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. 计算点积注意力得分 (Q * K^T) * scale
        # attn_scores: (B, num_heads, S_q, head_dim) @ (B, num_heads, head_dim, S_kv) -> (B, num_heads, S_q, S_kv)
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # 3. 施加注意力掩码 (Attention Mask)
        if attention_mask is not None:
            # attention_mask 形状为 (B, 1, S_q, S_kv) 或 (B, S_q, S_kv)
            # 值为 1 的保留，值为 0 的填充为极小负数 (-1e9)，Softmax 后权重趋近 0
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)

        # 4. Softmax 归一化为概率分布并与 V 加权求和
        attn_weights = torch.softmax(attn_scores, dim=-1)
        # out: (B, num_heads, S_q, S_kv) @ (B, num_heads, S_kv, head_dim) -> (B, num_heads, S_q, head_dim)
        out = torch.matmul(attn_weights, v)

        # 5. 多头拼接并过最终线性层
        # (B, num_heads, S_q, head_dim) -> (B, S_q, num_heads * head_dim) -> (B, S_q, D_q)
        out = out.transpose(1, 2).contiguous().view(B, S_q, self.hidden_size)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """
    两层 MLP 结构，标准 BERT/Transformer 配置
    """
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D_q) -> (B, S, intermediate_size) -> (B, S, D_q)
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class QFormerLayer(nn.Module):
    """
    Q-Former 的单个完整层
    包含:
      1. 自注意力机制 (Self-Attention with Residual & LayerNorm)
      2. 交叉注意力机制 (Cross-Attention, 专为 Query Tokens 设计)
      3. 前馈神经网络 (FFN with Residual & LayerNorm)
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

        # 2. 视觉特征 Cross-Attention (Key/Value 来自图像)
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
        详细的内部 Tensor 流转追踪:
        --------------------------------------------------------------------------
        hidden_states: (B, M + L, D_q)  [包含 M 个 Query 和 L 个 Text]
        image_embeds:  (B, N_img, D_img) [冻结 ViT 抽出的视觉 Token，例如 257 x 1408]
        num_query_tokens: M (通常为 32)
        --------------------------------------------------------------------------
        """
        # ==================== 1. Self-Attention 阶段 ====================
        # 输入: (B, M + L, D_q)
        # 受到 attention_mask 控制 (ITC/ITM/ITG)，实现各自的可见性逻辑
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = residual + self.self_attn(hidden_states, attention_mask=attention_mask)

        # ==================== 2. Cross-Attention 阶段 (核心关键点!) ====================
        # 规则: 仅有前 M 个 Query Token 允许去查询图像 image_embeds！
        # 后面的 Text Token (L 个) 保持原样，跳过 Cross-Attention。
        if self.has_cross_attention and image_embeds is not None:
            # 2.1 拆分出 Query 部分: (B, M, D_q)
            query_tokens = hidden_states[:, :num_query_tokens, :]

            # 2.2 残差 + LayerNorm + Cross-Attention
            residual_q = query_tokens
            query_norm = self.norm2(query_tokens)

            # Cross-Attention:
            #   Query: query_norm (B, M, D_q)
            #   Key/Value: image_embeds (B, N_img, D_img)
            # 输出: (B, M, D_q)
            query_out = residual_q + self.cross_attn(query_norm, context=image_embeds)

            # 2.3 拼接回序列:
            # 将更新后的 Query 与原封不动的 Text Token 拼回 (B, M + L, D_q)
            if hidden_states.shape[1] > num_query_tokens:
                text_tokens = hidden_states[:, num_query_tokens:, :]
                hidden_states = torch.cat([query_out, text_tokens], dim=1)
            else:
                # 只有纯 Query 输入（无 Text）的情况
                hidden_states = query_out

        # ==================== 3. Feed-Forward 阶段 ====================
        # 对所有的 M + L 个 tokens 均执行 FFN 映射
        residual = hidden_states
        hidden_states = self.norm3(hidden_states)
        hidden_states = residual + self.ffn(hidden_states)

        return hidden_states
