"""
lesson2_random_masking.py
=========================
【MAE 关卡 2】：Kaiming He 的双重 argsort 绝妙掩码算法 (Random Masking)

【为什么是双重 argsort？数学原理揭秘】:
假设我们有 4 个 Token: [A, B, C, D]，原始索引为 [0, 1, 2, 3]。
生成随机噪声:         [0.8, 0.2, 0.9, 0.1]
1. 第一次 argsort:
   噪声从小到大排序的原始索引:
   ids_shuffle = [3, 1, 0, 2]  (对应 [D, B, A, C])
   如果我们只要 50% 掩码，直接截取前两个: [D, B] 送进 Encoder！

2. 第二次 argsort:
   对 ids_shuffle 再做一次 argsort:
   ids_restore = torch.argsort(ids_shuffle)
   算出来结果为: [2, 1, 3, 0]！

   验证还原能力:
   如果我们拿着打乱后的序列 [D, B, A, C]，按照 ids_restore [2, 1, 3, 0] 取元素:
     第 2 位是 A
     第 1 位是 B
     第 3 位是 C
     第 0 位是 D
   结果恰好是 [A, B, C, D]！完美的无损原地复原！

测试命令:
  /Users/max/codebase/.ml/.venv/bin/python lesson2_random_masking.py
"""

import torch
from typing import Tuple


def random_masking(x: torch.Tensor, mask_ratio: float = 0.75) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    【输入】:
      - x: (B, N, D) 所有 Patch 的特征序列
      - mask_ratio: 掩码比例 (如 0.75，即遮蔽 75%，保留 25%)

    【输出】:
      - x_masked:    (B, len_keep, D) 仅剩 25% 长度的可见 Token
      - mask:        (B, N)           二值掩码 (0 为保留, 1 为被遮蔽)
      - ids_restore: (B, N)           复原原始空间顺序的索引矩阵
    """
    # =========================================================================
    # TODO 2.1: 请实现 random_masking
    # 步骤提示:
    # 1. 获取 B, N, D = x.shape
    # 2. 计算保留长度: len_keep = int(N * (1 - mask_ratio))
    # 3. 生成每个 patch 的随机均匀分布噪声:
    #    noise = torch.rand(B, N, device=x.device)  # (B, N)
    # 4. 第一次 argsort 获得打乱索引:
    #    ids_shuffle = torch.argsort(noise, dim=1)  # (B, N)
    # 5. 第二次 argsort 获得逆映射恢复索引:
    #    ids_restore = torch.argsort(ids_shuffle, dim=1)  # (B, N)
    # 6. 取前 len_keep 个索引:
    #    ids_keep = ids_shuffle[:, :len_keep]       # (B, len_keep)
    # 7. 利用 torch.gather 抽取可见 Token:
    #    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    # 8. 生成二值 mask 矩阵:
    #    mask = torch.ones([B, N], device=x.device)
    #    mask[:, :len_keep] = 0
    #    mask = torch.gather(mask, dim=1, index=ids_restore)  # 恢复原始空间对应
    # 9. 返回 x_masked, mask, ids_restore
    # =========================================================================
    raise NotImplementedError("TODO 2.1 尚未实现！请实现 random_masking")


# ==============================================================================
# 单元测试验证
# ==============================================================================
def run_test():
    print("\n========== 开始测试 MAE 关卡 2 ==========")
    B, N, D = 2, 196, 768
    mask_ratio = 0.75
    len_keep = int(N * (1 - mask_ratio))  # 49

    # 1. 构造一个能自验证索引位置的特殊张量:
    # 每个 Token 的所有维度上的值等于其原本的位置编号 [0, 1, 2, ..., N-1]
    tokens = torch.arange(N).unsqueeze(0).repeat(B, 1).unsqueeze(-1).repeat(1, 1, D).float()

    x_masked, mask, ids_restore = random_masking(tokens, mask_ratio=mask_ratio)

    # 2. 检查输出形状
    assert x_masked.shape == (B, len_keep, D), f"x_masked 形状错误: 期望 {(B, len_keep, D)}, 实际 {x_masked.shape}"
    assert mask.shape == (B, N), f"mask 形状错误: 期望 {(B, N)}, 实际 {mask.shape}"
    assert ids_restore.shape == (B, N), f"ids_restore 形状错误: 期望 {(B, N)}, 实际 {ids_restore.shape}"
    print(f"✅ 维度检查通过: x_masked 仅保留 25% 长度 ({len_keep}/{N})")

    # 3. 检查 Mask 统计量 (必须恰好遮蔽 75% 的 token)
    num_masked = mask[0].sum().item()
    assert num_masked == (N - len_keep), f"被遮蔽数量错误: 期望 {N - len_keep}, 实际 {num_masked}"
    print(f"✅ 掩码比例检查通过: 遮蔽率 {num_masked / N:.2%}")

    # 4. 【最硬核断言】：用 ids_restore 完美复原序列
    # 构造假想的 mask_tokens (值为 -999) 拼接到 x_masked 后面
    num_masked = N - len_keep
    fake_mask_tokens = torch.full((B, num_masked, D), -999.0)
    shuffled_all = torch.cat([x_masked, fake_mask_tokens], dim=1)  # (B, N, D)

    # 用 ids_restore 做 gather 还原
    restored = torch.gather(shuffled_all, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))

    # 断言: 在那些 mask == 0 的可见位置上，数值必须 100% 精确回到原位！
    visible_positions = (mask == 0)
    original_visible_vals = tokens[visible_positions]
    restored_visible_vals = restored[visible_positions]
    assert torch.allclose(original_visible_vals, restored_visible_vals), "ids_restore 还原后可见 Token 位置不吻合！"
    print("✅ 双重 argsort 原地还原机制断言通过！")
    print("🎉 恭喜！关卡 2 全部挑战成功！\n")


if __name__ == "__main__":
    run_test()
