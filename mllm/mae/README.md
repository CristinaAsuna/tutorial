# 从零手写 MAE (Masked Autoencoders) 硬核闯关实战

> 共享基础包：先执行 `python3 -m pip install -e "../../general_utils/edu_core[dev]"`。本目录的 `recipe/` 是真实 ImageNet 训练配方合同；本 README 其余代码仍是 CPU 机制教学实现。

欢迎来到 **MAE (Masked Autoencoders Are Scalable Vision Learners, Kaiming He et al., CVPR 2022)** 的硬核编程实战！

如果说写普通 Transformer 只是在调用矩阵乘法，那么 **手写 MAE 则是对 PyTorch 高阶张量操作（高维重排、双重 argsort、gather/scatter 索引对齐、2D 正余弦位置编码）最顶级的磨刀石**。

---

## 目录与闯关路线

```text
mae/
├── README.md                  # 核心数学与张量维度图解（本文档）
├── reference_mae.py           # [标准参考答案] 完整端到端可运行的 MAE 实现
├── lesson1_patchify.py        # [关卡 1] 硬核张量重排：纯算子实现 patchify 与 unpatchify
├── lesson2_random_masking.py  # [关卡 2] 绝妙算法：双重 argsort 实现 75% 掩码打乱与还原
├── lesson3_pos_embed.py       # [关卡 3] 数学几何：手写 2D Sin-Cos 正余弦位置编码
├── lesson4_mae_model.py       # [关卡 4] 非对称架构：轻量 Decoder、Mask Token 插入与 Masked MSE Loss
└── run_mae_demo.py            # [实战演练] 合成图像前向重构与可视化验证
```

---

## 核心张量维度速查 (Cheat-Sheet)

| 符号 | 含义 | 典型值 (ViT-Base) |
| :--- | :--- | :--- |
| `B` | Batch Size (批次大小) | 4 |
| `C` | 图像通道数 | 3 |
| `H, W` | 图像原始高度与宽度 | 224, 224 |
| `P` | Patch 大小 (如 16x16) | 16 |
| `h, w` | Patch 网格高宽 ($H/P, W/P$) | 14, 14 |
| `N` | Patch 总数量 ($h \times w$) | **196** |
| `D_enc` | 编码器隐层维度 | **768** |
| `D_dec` | 解码器隐层维度 | **512** (非对称设计，解码器轻量化) |
| `mask_ratio` | 掩码比例 | **0.75** (遮蔽 75%，保留 25%) |
| `len_keep` | 编码器实际计算的 Token 数 | $196 \times (1 - 0.75) =$ **49** |

---

## 为什么说 MAE 是最锻炼硬核编码能力的项目？

1. **零冗余计算的 Patchify / Unpatchify**：如何在不用 `einops` 的情况下，用纯原生的 `view` 和 `permute` 将 4D 图像切成 2D Patch，并在预测后无损拼回原图？
2. **Kaiming He 的双重 `argsort` 奇技淫巧**：如何用两行 `argsort`，在完全不构建全尺寸 Mask 矩阵的情况下，完成**打乱抽取可见 Token $\to$ 送入 Encoder $\to$ 用 `ids_restore` 原地复原位置送入 Decoder**？
3. **2D 网格正余弦编码**：如何分别在高度轴和宽度轴上推导 1D Sin-Cos 编码，并通过笛卡尔积（Cartesian Product）融合成 2D 坐标编码？
4. **单 Patch 均值方差归一化损失 (Normalized MSE)**：为什么在 Patch 级别做零均值单位方差归一化可以大幅提升自监督表征质量？

---

## 参考实现约定

- `reference_mae.py` 是可运行的参考答案；`lesson1_*.py` 至 `lesson4_*.py` 保留 TODO，预期在完成前抛出 `NotImplementedError`。`run_mae_demo.py` 只验证参考答案。
- 这是机制教学实现，不包含真实数据集、训练循环、优化器、分布式预训练或下游微调。
- 输入图像必须为构造模型时指定的正方形尺寸；`patchify`/`unpatchify` 也要求正方形 patch 网格，以便清晰展示 token 与空间位置的一一对应。
- `mask_ratio` 的有效范围是 `(0, 1]`。MAE 的 masked MSE 只在 `mask == 1` 的 patch 上计算；`mask_ratio=0` 没有监督目标，因此被明确拒绝。
- 参考实现使用固定 2D sin-cos 位置编码，并采用官方 MAE 风格的 Xavier/LayerNorm/CLS/mask-token 初始化；因此初始 loss 数值会随实现版本变化，但张量数据流不变。
