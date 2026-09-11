"""
attention_masks.py
===================
BLIP-2 Q-Former 的核心精髓：三大注意力掩码 (Attention Masks)

在 Q-Former 中，Query Tokens (长度 M) 和 Text Tokens (长度 L) 被拼成一个统一的长序列:
    Combined Sequence = [Query_1, ..., Query_M, Text_1, ..., Text_L]
    总长度 S = M + L

由于两者共享同一个 Self-Attention 权重，我们必须通过注意力矩阵 (S, S) 精确控制：
“谁能看见谁，谁不能看见谁”。

掩码矩阵形状定义:
    Matrix[i, j] 表示位置 i 处的 Token 是否能看 (Attend to) 位置 j 处的 Token。
    1 表示可见 (可以计算注意力), 0 表示遮蔽 (Attention Weight -> -inf)

矩阵被划分为 4 个象限:
                |  Query (0..M)  |   Text (M..M+L)  |
   -------------+----------------+------------------+
   Query (0..M) |    Q -> Q      |      Q -> T      |
   -------------+----------------+------------------+
   Text (M..M+L)|    T -> Q      |      T -> T      |
   -------------+----------------+------------------+
"""

import torch


def create_itc_mask(batch_size: int, num_query_tokens: int, text_attention_mask: torch.Tensor) -> torch.Tensor:
    """
    1. ITC Mask (Image-Text Contrastive, 图文对比学习掩码)

    【设计目的】:
      - 图像 Query 之间可以互看，用于汇聚全局图像语义。
      - 文本 Text 之间可以互看，用于提取文本语义 (类似于标准 BERT)。
      - Query 和 Text 之间【绝对互不相见】！避免在计算对比度时信息泄露。

    【结构可视化 (M=3, L=3)】:
        Q  Q  Q  |  T  T  T
      Q [1, 1, 1, |  0, 0, 0]
      Q [1, 1, 1, |  0, 0, 0]
      Q [1, 1, 1, |  0, 0, 0]
      ---+---------+---------
      T [0, 0, 0, |  1, 1, 1]
      T [0, 0, 0, |  1, 1, 1]
      T [0, 0, 0, |  1, 1, 1]

    【Tensor 变化】:
      输入:
        text_attention_mask: (B, L) 原始文本 padding mask (1 为真实 token, 0 为 pad)
      输出:
        mask: (B, 1, M + L, M + L) 广播用于 Multi-Head Attention
    """
    B, L = text_attention_mask.shape
    M = num_query_tokens
    total_len = M + L

    # 初始化全 0 矩阵: (B, total_len, total_len)
    mask = torch.zeros((B, total_len, total_len), device=text_attention_mask.device, dtype=torch.long)

    # 1. Q -> Q: Query 互见 (取值 1)
    mask[:, :M, :M] = 1

    # 2. Q -> T: 0 (保持为 0)
    # 3. T -> Q: 0 (保持为 0)

    # 4. T -> T: Text 互见，但需要考虑 Text 自身的 padding
    # text_attention_mask[:, None, :] 是 (B, 1, L)，按行广播表示只要目标 token 不是 pad 就可见
    # text_attention_mask[:, :, None] 是 (B, L, 1)，自己的 pad token 也不去关注别人
    text_self_mask = text_attention_mask.unsqueeze(1) * text_attention_mask.unsqueeze(2)  # (B, L, L)
    mask[:, M:, M:] = text_self_mask

    # 扩展维度为 (B, 1, total_len, total_len) 方便和 (B, num_heads, total_len, total_len) 运算
    return mask.unsqueeze(1)


def create_itm_mask(batch_size: int, num_query_tokens: int, text_attention_mask: torch.Tensor) -> torch.Tensor:
    """
    2. ITM Mask (Image-Text Matching, 图文匹配二分类掩码)

    【设计目的】:
      - 允许深度的跨模态全双向交互（Bi-directional Attention）。
      - Query 能看 Text，Text 也能看 Query。
      - 输出层取 Query 的表征，送入二分类线性头，判断当前“图-文”是不是配对的。

    【结构可视化 (M=3, L=3)】:
        Q  Q  Q  |  T  T  T
      Q [1, 1, 1, |  1, 1, 1]
      Q [1, 1, 1, |  1, 1, 1]
      Q [1, 1, 1, |  1, 1, 1]
      ---+---------+---------
      T [1, 1, 1, |  1, 1, 1]
      T [1, 1, 1, |  1, 1, 1]
      T [1, 1, 1, |  1, 1, 1]

    【Tensor 变化】:
      输出: (B, 1, M + L, M + L)
    """
    B, L = text_attention_mask.shape
    M = num_query_tokens
    total_len = M + L

    # 构建一维序列的有效性标记: Query 全有效 (1)，Text 根据 padding mask 判断 (1 或 0)
    query_valid = torch.ones((B, M), device=text_attention_mask.device, dtype=torch.long)
    seq_valid = torch.cat([query_valid, text_attention_mask], dim=1)  # (B, total_len)

    # 构造全交互掩码: (B, total_len, total_len)
    # 只要被关注的目标 token 是有效的，就可以被关注
    mask = seq_valid.unsqueeze(1).repeat(1, total_len, 1)  # (B, total_len, total_len)

    return mask.unsqueeze(1)


def create_itg_mask(batch_size: int, num_query_tokens: int, text_attention_mask: torch.Tensor) -> torch.Tensor:
    """
    3. ITG Mask (Image-Grounded Text Generation, 条件文本因果生成掩码)

    【设计目的】:
      - 图像描述生成（Captioning）。
      - Query 充当 Prefix 前缀条件，Query 内部全双向可见。
      - Query【绝对不能看】Text，否则会产生未来信息泄露！
      - Text 可以看到所有 Query，作为生成的视觉上下文条件。
      - Text 内部是因果因果掩码 (Causal Mask / 下三角)，只能看前序 token，不能看未来 token。

    【结构可视化 (M=3, L=3)】:
        Q  Q  Q  |  T1 T2 T3
      Q [1, 1, 1, |  0, 0, 0]   <-- Query 不能看 Text
      Q [1, 1, 1, |  0, 0, 0]
      Q [1, 1, 1, |  0, 0, 0]
      ---+---------+---------
      T1[1, 1, 1, |  1, 0, 0]   <-- T1 看到全部 Query，和自己
      T2[1, 1, 1, |  1, 1, 0]   <-- T2 看到全部 Query，和 T1、T2
      T3[1, 1, 1, |  1, 1, 1]   <-- T3 看到全部 Query，和 T1、T2、T3

    【Tensor 变化】:
      输出: (B, 1, M + L, M + L)
    """
    B, L = text_attention_mask.shape
    M = num_query_tokens
    total_len = M + L

    mask = torch.zeros((B, total_len, total_len), device=text_attention_mask.device, dtype=torch.long)

    # 1. Q -> Q: 互见
    mask[:, :M, :M] = 1

    # 2. Q -> T: 0 (不能偷看未来文本)

    # 3. T -> Q: Text 可以看所有的 Query (取值 1，但如果当前 text 本身是 pad 则由 loss 掩蔽)
    mask[:, M:, :M] = 1

    # 4. T -> T: 因果因果下三角掩码 + Text padding mask
    causal_mask = torch.tril(torch.ones((L, L), device=text_attention_mask.device, dtype=torch.long))  # (L, L)
    causal_mask = causal_mask.unsqueeze(0).repeat(B, 1, 1)  # (B, L, L)

    # 同时过滤掉 pad token
    text_pad_mask = text_attention_mask.unsqueeze(1)  # (B, 1, L)
    causal_text_mask = causal_mask * text_pad_mask
    mask[:, M:, M:] = causal_text_mask

    return mask.unsqueeze(1)
