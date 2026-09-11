"""
lesson2_attention_masks.py
==========================
【关卡 2】：三大注意力掩码 (Attention Masks)

在 Q-Former 中，Query (M 个) 与 Text (L 个) 被拼接为一个总长度为 M + L 的统一序列送入 Self-Attention。
我们需要用 (M+L, M+L) 的 2D 矩阵来控制谁能看谁:
    - 值为 1: 可见 (参与注意力)
    - 值为 0: 遮蔽 (权重为 -inf)

本文件需要你实现 3 种 Mask 生成函数:
1. create_itc_mask: 图文对比掩码 (Q 与 T 互相隔离)
2. create_itm_mask: 图文匹配掩码 (Q 与 T 完全双向交互)
3. create_itg_mask: 条件生成掩码 (Q 是 Prefix, T 内部因果下三角)

测试命令:
  /Users/max/codebase/.ml/.venv/bin/python lesson2_attention_masks.py
"""

import torch


# ==============================================================================
# 函数 1: ITC Mask (Image-Text Contrastive)
# ==============================================================================
def create_itc_mask(batch_size: int, num_query_tokens: int, text_attention_mask: torch.Tensor) -> torch.Tensor:
    """
    【设计意图】:
      - Query 负责看 Query 提取图像全局特征
      - Text 负责看 Text 提取文本特征 (按 padding mask)
      - Query 和 Text 之间【绝对不可见】(防止对比学习时信息偷窥)

    【矩阵示意图】:
        Q   Q   |  T   T
      Q [1,  1,  |  0,  0]
      Q [1,  1,  |  0,  0]
      ---+-------+-------
      T [0,  0,  |  1,  1]
      T [0,  0,  |  1,  1]

    【输入】:
      - batch_size: B
      - num_query_tokens: M
      - text_attention_mask: (B, L) 原始文本 padding mask (1 为有效，0 为 pad)
    【输出】:
      - mask: (B, 1, M + L, M + L)
    """
    # =========================================================================
    # TODO 2.1: 请构造 ITC Mask
    # 步骤提示:
    # 1. total_len = M + L
    # 2. 初始化全 0 矩阵: mask = torch.zeros((B, total_len, total_len), device=..., dtype=torch.long)
    # 3. 将左上角 Q -> Q 区域设为 1: mask[:, :M, :M] = 1
    # 4. 右下角 T -> T 区域: 利用 text_attention_mask 的行和列做外积 (考虑 pad token):
    #    text_self_mask = text_attention_mask.unsqueeze(1) * text_attention_mask.unsqueeze(2)  # (B, L, L)
    #    mask[:, M:, M:] = text_self_mask
    # 5. 返回扩展维度的 mask.unsqueeze(1) -> (B, 1, total_len, total_len)
    # =========================================================================
    raise NotImplementedError("TODO 2.1 尚未实现！请实现 create_itc_mask")


# ==============================================================================
# 函数 2: ITM Mask (Image-Text Matching)
# ==============================================================================
def create_itm_mask(batch_size: int, num_query_tokens: int, text_attention_mask: torch.Tensor) -> torch.Tensor:
    """
    【设计意图】:
      - 允许深度的全双向多模态交互 (Bi-directional)。
      - 只要目标 Token 是有效 Token (非 Pad)，任何 Token 都可以看到它。

    【输入输出】:
      - 输出: (B, 1, M + L, M + L)
    """
    # =========================================================================
    # TODO 2.2: 请构造 ITM Mask
    # 步骤提示:
    # 1. total_len = M + L
    # 2. Query 全为有效 token (1): query_valid = torch.ones((B, M), device=..., dtype=torch.long)
    # 3. 沿序列拼接有效性向量: seq_valid = torch.cat([query_valid, text_attention_mask], dim=1) -> (B, total_len)
    # 4. 沿行方向重复，每个 token 都能看到 seq_valid 中所有为 1 的位置:
    #    mask = seq_valid.unsqueeze(1).repeat(1, total_len, 1)  # (B, total_len, total_len)
    # 5. 返回 mask.unsqueeze(1)
    # =========================================================================
    raise NotImplementedError("TODO 2.2 尚未实现！请实现 create_itm_mask")


# ==============================================================================
# 函数 3: ITG Mask (Image-Grounded Text Generation)
# ==============================================================================
def create_itg_mask(batch_size: int, num_query_tokens: int, text_attention_mask: torch.Tensor) -> torch.Tensor:
    """
    【设计意图】:
      - 文本自回归生成。
      - Query 内部全互见，作为生成的视觉前缀条件。
      - Query【绝对不能看】Text (右上方为 0，防止偷看未来词)。
      - Text 可以看所有 Query (左下方全为 1)。
      - Text 内部是因果因果掩码 (右下方是下三角矩阵)。

    【矩阵示意图】:
        Q   Q   |  T1  T2
      Q [1,  1,  |  0,  0]   <-- Q 不看 T
      Q [1,  1,  |  0,  0]
      ---+-------+-------
      T1[1,  1,  |  1,  0]   <-- T1 看所有 Q，看自身
      T2[1,  1,  |  1,  1]   <-- T2 看所有 Q，看 T1 和自身
    """
    # =========================================================================
    # TODO 2.3: 请构造 ITG Mask
    # 步骤提示:
    # 1. total_len = M + L, 初始化全 0 矩阵 (B, total_len, total_len)
    # 2. Q -> Q 设为 1: mask[:, :M, :M] = 1
    # 3. T -> Q 设为 1: mask[:, M:, :M] = 1
    # 4. T -> T 构造下三角矩阵:
    #    causal_mask = torch.tril(torch.ones((L, L), device=...)).unsqueeze(0).repeat(B, 1, 1)
    #    结合 text_attention_mask.unsqueeze(1) 过滤 pad
    #    mask[:, M:, M:] = causal_mask * text_attention_mask.unsqueeze(1)
    # 5. 返回 mask.unsqueeze(1)
    # =========================================================================
    raise NotImplementedError("TODO 2.3 尚未实现！请实现 create_itg_mask")


# ==============================================================================
# 单元测试验证
# ==============================================================================
def run_test():
    print("\n========== 开始测试关卡 2 ==========")
    B, M, L = 2, 3, 3
    # 构造一个 pad mask，最后一个 token 为 0 (Pad)
    text_pad_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    # 1. 测试 ITC
    itc_mask = create_itc_mask(B, M, text_pad_mask)
    assert itc_mask.shape == (B, 1, M + L, M + L), f"ITC 形状不符: {itc_mask.shape}"
    # 断言 Q 不能看 T
    assert (itc_mask[:, 0, :M, M:] == 0).all(), "ITC Mask 中 Query 不应看到 Text！"
    # 断言 T 不能看 Q
    assert (itc_mask[:, 0, M:, :M] == 0).all(), "ITC Mask 中 Text 不应看到 Query！"
    print("✅ TODO 2.1 (create_itc_mask) 通过测试！")

    # 2. 测试 ITM
    itm_mask = create_itm_mask(B, M, text_pad_mask)
    assert itm_mask.shape == (B, 1, M + L, M + L), f"ITM 形状不符: {itm_mask.shape}"
    # 断言首个样本中 Q 可以看有效的 T
    assert itm_mask[0, 0, 0, M] == 1, "ITM Mask 中 Query 应该可以看到有效 Text！"
    assert itm_mask[0, 0, 0, -1] == 0, "ITM Mask 中不应看到 Pad Token！"
    print("✅ TODO 2.2 (create_itm_mask) 通过测试！")

    # 3. 测试 ITG
    itg_mask = create_itg_mask(B, M, text_pad_mask)
    assert itg_mask.shape == (B, 1, M + L, M + L), f"ITG 形状不符: {itg_mask.shape}"
    # 断言 Q 绝不能看 T
    assert (itg_mask[:, 0, :M, M:] == 0).all(), "ITG Mask 中 Query 绝不能看到未来的 Text！"
    # 断言 T 可以看 Q
    assert (itg_mask[:, 0, M:, :M] == 1).all(), "ITG Mask 中 Text 必须能看到所有 Query 前缀！"
    # 断言因果下三角特性: T1 不能看 T2
    assert itg_mask[0, 0, M, M + 1] == 0, "ITG Mask 中 T1 绝不能看到 T2 (未来词)！"
    print("✅ TODO 2.3 (create_itg_mask) 通过测试！")
    print("🎉 恭喜！关卡 2 全部挑战成功！\n")


if __name__ == "__main__":
    run_test()
