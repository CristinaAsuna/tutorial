"""
lesson3_pos_embed.py
====================
【MAE 关卡 3】：2D 正余弦几何位置编码 (2D Sin-Cos Positional Embeddings)

【为什么 MAE 不用可学习的位置编码，而是用固定的 2D Sin-Cos？】:
1. MAE 中大量的 Patch 被完全抹去 (75%)，固定的正余弦编码天然具有几何先验，不会在遮蔽训练中过拟合或退化。
2. 2D 网格分解: 将总维度 D 对半拆分:
   - D/2 维度负责对 Y 轴 (垂直高度行坐标) 进行 1D 正余弦编码
   - D/2 维度负责对 X 轴 (水平宽度列坐标) 进行 1D 正余弦编码
   - 两者拼接为完整的 (H * W, D) 2D 坐标表征！

测试命令:
  /Users/max/codebase/.ml/.venv/bin/python lesson3_pos_embed.py
"""

import numpy as np
import torch


# ==============================================================================
# 任务 1: 1D 正余弦编码
# ==============================================================================
def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """
    【输入】:
      - embed_dim: 编码维度 (必须为偶数)
      - pos: 一维坐标数组 (如 [0, 1, 2, ..., M-1])，形状任意，会展平为 (M,)

    【输出】:
      - emb: (M, embed_dim)
    """
    assert embed_dim % 2 == 0, "embed_dim 必须为偶数"

    # =========================================================================
    # TODO 3.1: 请实现 1D 正余弦编码
    # 步骤提示:
    # 1. 构造角频率衰减因子 omega (长度为 embed_dim // 2):
    #    omega = np.arange(embed_dim // 2, dtype=np.float32)
    #    omega /= embed_dim / 2.
    #    omega = 1. / (10000**omega)  # 形状: (D/2,)
    # 2. 计算点积网格 out = pos.reshape(-1) 与 omega 的外积:
    #    out = np.einsum('m,d->md', pos.reshape(-1), omega)  # (M, D/2)
    # 3. 分别计算 sin 和 cos:
    #    emb_sin = np.sin(out)
    #    emb_cos = np.cos(out)
    # 4. 在特征维度拼接:
    #    emb = np.concatenate([emb_sin, emb_cos], axis=1)   # (M, D)
    # 5. 返回 emb
    # =========================================================================
    raise NotImplementedError("TODO 3.1 尚未实现！请实现 get_1d_sincos_pos_embed_from_grid")


# ==============================================================================
# 任务 2: 2D 网格组合位置编码
# ==============================================================================
def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = False) -> np.ndarray:
    """
    【输入】:
      - embed_dim: 总特征维度 (如 768)
      - grid_size: 网格大小 (如 14，对应 14x14 = 196 个 patch)
      - cls_token: 是否在开头拼接一个 0 向量代表 [CLS] 的占位位置编码

    【输出】:
      - pos_embed: (grid_size * grid_size, embed_dim) 若含 cls_token 则第一维为 +1
    """
    # =========================================================================
    # TODO 3.2: 请实现 2D 正余弦编码
    # 步骤提示:
    # 1. 生成高宽坐标网格:
    #    grid_h = np.arange(grid_size, dtype=np.float32)
    #    grid_w = np.arange(grid_size, dtype=np.float32)
    #    grid = np.meshgrid(grid_w, grid_h)  # 包含网格 x 和 y 坐标
    # 2. 堆叠并重排:
    #    grid = np.stack(grid, axis=0)       # (2, grid_size, grid_size)
    #    grid = grid.reshape([2, 1, grid_size, grid_size])
    # 3. 分配维度: 高度与宽度各分 embed_dim // 2:
    #    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    #    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    # 4. 拼接成 2D 编码:
    #    pos_embed = np.concatenate([emb_h, emb_w], axis=1)                  # (H*W, D)
    # 5. 如果 cls_token 为 True，在头部拼一行全 0 向量:
    #    pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    # 6. 返回 pos_embed
    # =========================================================================
    raise NotImplementedError("TODO 3.2 尚未实现！请实现 get_2d_sincos_pos_embed")


# ==============================================================================
# 单元测试验证
# ==============================================================================
def run_test():
    print("\n========== 开始测试 MAE 关卡 3 ==========")
    D = 768
    grid_size = 14
    num_patches = grid_size * grid_size  # 196

    # 1. 测试 1D 编码
    pos_1d = np.arange(10)
    emb_1d = get_1d_sincos_pos_embed_from_grid(64, pos_1d)
    assert emb_1d.shape == (10, 64), f"1D 编码形状错误: {emb_1d.shape}"
    print("✅ TODO 3.1 (1D 正余弦) 通过测试！")

    # 2. 测试 2D 编码 (无 cls_token)
    pos_2d = get_2d_sincos_pos_embed(D, grid_size, cls_token=False)
    assert pos_2d.shape == (num_patches, D), f"2D 编码形状错误: {pos_2d.shape}"

    # 几何连续性验证: 相邻网格的余弦相似度应该很高，对角线远端的余弦相似度应该较低
    t_pos = torch.from_numpy(pos_2d)
    sim_adjacent = torch.cosine_similarity(t_pos[0:1], t_pos[1:2]).item()
    sim_far = torch.cosine_similarity(t_pos[0:1], t_pos[-1:]).item()
    assert sim_adjacent > sim_far, "几何距离近的 Patch 余弦相似度应当高于远端 Patch！"
    print(f"✅ 几何空间度量正常 (相邻相似度: {sim_adjacent:.3f} > 远端相似度: {sim_far:.3f})")

    # 3. 测试含 cls_token
    pos_2d_cls = get_2d_sincos_pos_embed(D, grid_size, cls_token=True)
    assert pos_2d_cls.shape == (num_patches + 1, D)
    assert np.allclose(pos_2d_cls[0], 0.0), "[CLS] Token 初始位置编码必须为 0！"
    print("✅ TODO 3.2 (2D 正余弦网格) 通过测试！")
    print("🎉 恭喜！关卡 3 全部挑战成功！\n")


if __name__ == "__main__":
    run_test()
