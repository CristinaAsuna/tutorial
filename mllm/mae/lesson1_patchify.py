"""
lesson1_patchify.py
===================
【MAE 关卡 1】：图像切片与无损还原 (Patchify 与 Unpatchify)

在视觉 Transformer 中，图像不是以像素矩阵参与计算的，而是切成一块块小 Patch (如 16x16)。
但在 MAE 中，因为 Decoder 最终要预测每个 Patch 的原始像素值，
我们必须写出【完全可逆、零数值损耗】的两种变换:
  - patchify:   (B, 3, H, W)       ---> (B, N, P * P * 3)
  - unpatchify: (B, N, P * P * 3)  ---> (B, 3, H, W)

测试命令:
  /Users/max/codebase/.ml/.venv/bin/python lesson1_patchify.py
"""

import torch


# ==============================================================================
# 任务 1: Patchify (4D 图像 -> 2D 序列)
# ==============================================================================
def patchify(imgs: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
    """
    【设计思路】:
      不用任何第三方库 (如 einops)，纯粹利用 PyTorch 原生的 reshape 与 permute。

    【Tensor 维度推导】:
      输入 imgs: (B, C, H, W)，其中假设 H == W
      令 h = H // p, w = W // p (即网格尺寸，如 224//16 = 14)
      1. reshape 为 6D:
         (B, C, h, p, w, p)
      2. permute 调整轴序:
         我们要把同一个 Patch 内部的空间像素 (p, p) 和颜色通道 C 放到连续内存中！
         目标轴序: (B, h, w, p, p, C)
      3. 最后 reshape 展平成 3D 序列:
         (B, h * w, p * p * C) -> 即 (B, N, 768)
    """
    # =========================================================================
    # TODO 1.1: 请实现 patchify
    # 步骤提示:
    # 1. 获取 B, C, H, W = imgs.shape, p = patch_size
    # 2. 计算 h = w = H // p
    # 3. x = imgs.reshape(B, C, h, p, w, p)
    # 4. x = x.permute(0, 2, 4, 3, 5, 1)  # 变为 (B, h, w, p, p, C)
    # 5. patches = x.reshape(B, h * w, p * p * C)
    # 6. 返回 patches
    # =========================================================================
    raise NotImplementedError("TODO 1.1 尚未实现！请实现 patchify")


# ==============================================================================
# 任务 2: Unpatchify (2D 序列 -> 4D 图像)
# ==============================================================================
def unpatchify(patches: torch.Tensor, patch_size: int = 16, channels: int = 3) -> torch.Tensor:
    """
    【设计思路】:
      patchify 的精确逆运算！将预测出的每个 Patch 的像素无缝拼回原始图像。

    【Tensor 维度推导】:
      输入 patches: (B, N, p * p * C)
      令 h = w = int(sqrt(N)) (网格宽高)
      1. reshape 为 6D:
         (B, h, w, p, p, C)
      2. permute 逆向恢复轴序:
         将通道 C 提回前面，空间网格与局部像素交替排列:
         (B, C, h, p, w, p)
      3. 最后 reshape 拼回 4D 图像:
         (B, C, h * p, w * p) -> 即 (B, 3, H, W)
    """
    # =========================================================================
    # TODO 1.2: 请实现 unpatchify
    # 步骤提示:
    # 1. 获取 B, N, D = patches.shape, p = patch_size, C = channels
    # 2. 计算 h = w = int(N ** 0.5)
    # 3. x = patches.reshape(B, h, w, p, p, C)
    # 4. x = x.permute(0, 5, 1, 3, 2, 4)  # 变为 (B, C, h, p, w, p)
    # 5. imgs = x.reshape(B, C, h * p, w * p)
    # 6. 返回 imgs
    # =========================================================================
    raise NotImplementedError("TODO 1.2 尚未实现！请实现 unpatchify")


# ==============================================================================
# 单元测试验证
# ==============================================================================
def run_test():
    print("\n========== 开始测试 MAE 关卡 1 ==========")
    B, C, H, W = 4, 3, 224, 224
    P = 16
    N = (H // P) * (W // P)  # 196
    patch_dim = P * P * C    # 768

    # 构造随机图像
    original_imgs = torch.randn(B, C, H, W)

    # 1. 验证 Patchify 形状
    patches = patchify(original_imgs, patch_size=P)
    assert patches.shape == (B, N, patch_dim), f"Patchify 输出形状错误: 期望 {(B, N, patch_dim)}, 实际 {patches.shape}"
    print(f"✅ TODO 1.1 (patchify) 形状通过: {patches.shape}")

    # 2. 验证 Unpatchify 形状
    recovered_imgs = unpatchify(patches, patch_size=P, channels=C)
    assert recovered_imgs.shape == (B, C, H, W), f"Unpatchify 输出形状错误: 期望 {(B, C, H, W)}, 实际 {recovered_imgs.shape}"
    print(f"✅ TODO 1.2 (unpatchify) 形状通过: {recovered_imgs.shape}")

    # 3. 【最严苛测试】数值无损完全重构断言 (误差必须为 0)
    max_diff = (original_imgs - recovered_imgs).abs().max().item()
    assert torch.allclose(original_imgs, recovered_imgs, atol=1e-6), f"重构图像与原图数值不一致！最大误差: {max_diff}"
    print(f"✅ 无损重构验证通过！原图与还原图最大误差: {max_diff:.2e}")
    print("🎉 恭喜！关卡 1 全部挑战成功！\n")


if __name__ == "__main__":
    run_test()
