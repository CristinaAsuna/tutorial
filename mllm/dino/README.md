# 从零手写 DINOv2 核心：DINO + iBOT

> 共享基础包：先执行 `python3 -m pip install -e "../../general_utils/edu_core[dev]"`。顶层内容是 CPU toy；`recipe/` 记录真实 DINO/iBOT 训练所需条件及 DINOv2 不可严格公开复现的边界。

这是一个面向理解的数据流教学实现：不下载数据集、不依赖 torchvision，也不复刻完整生产版 DINOv2。它自写一个最小 ViT，跑通多裁剪自蒸馏、EMA teacher、centering / temperature，以及 iBOT 的 masked patch prediction。

```text
dino/
├── reference_dinov2.py          # 完整可运行参考解
├── multicrop.py                 # 仅 PyTorch 的随机裁剪、resize、patch mask
├── lesson1_vit.py                # TODO：最小 ViT 与位置编码插值
├── lesson2_multicrop_masking.py  # TODO：multi-crop 与 patch mask
├── lesson3_dino_loss.py          # TODO：DINO cross-view loss / center
├── lesson4_ibot_teacher_student.py # TODO：iBOT patch loss / EMA
└── run_dinov2_demo.py            # 一个 CPU 合成数据训练 step
```

## 快速运行

```bash
cd /Users/max/codebase/scratch/tutorial/mllm/dino
python run_dinov2_demo.py
```

demo 固定随机种子、用两张 `40×40` 随机 RGB 图像，在 CPU 上完成：student 前向与反传、optimizer step、teacher EMA、CLS center 与 patch center 更新。它还断言 student 有梯度和更新、teacher 永远 eval/冻结但会随 EMA 改变。

## 一次训练 step 的数据流

```text
images [B,3,H,W]
  └─ multi-crop ──> 2 global [B,3,32,32] + 4 local [B,3,16,16]
                      │                         │
teacher (eval,frozen)│                         └─ student sees all 6 crops
sees global only     │                              global patches get mask token
  └─ CLS [B,K], patch [B,16,K]                    └─ CLS [B,K], patch [B,N,K]
       │                                    │
       ├─ DINO: teacher global CLS targets ──┘  cross-view CLS loss
       └─ iBOT: same-view unmasked patch target ─ masked global patches only
```

`B` 是 batch size；`P=8` 时，`32×32` global crop 有 `N=16` 个 patch，`16×16` local crop 有 `N=4` 个 patch；`D` 是 ViT embedding dimension，`K` 是 prototype 数量（demo 为 32）。`DINOHead` 对 CLS 和 patch token 共用同一个 MLP/prototype layer，所以两者最终都是 logits `[..., K]`。

## 关键算法语义

- DINO：teacher 的两个 global CLS 分布是 soft target；student 全部六个 CLS 都参与匹配。仅跳过 global-0 对 global-0、global-1 对 global-1 的同视图配对。
- iBOT：teacher 对相同 global view 的**未遮蔽** patch 给 soft target；student 输入中相应 patch 被替换为可学习的 `mask_token`。损失严格只取 `mask=True` 的位置。
- Teacher：由 student 初始化，始终 `eval()` 且 `requires_grad=False`。每一次 `optimizer.step()` 后执行 `teacher = momentum * teacher + (1-momentum) * student`。
- Center：`center` 与 `patch_center` 是 buffer，不参与优化；分别取本 batch teacher CLS/patch logits 的均值后做 EMA，进入 teacher softmax 前减去它们。teacher temperature 可从较高值 warm up 到目标值。

## 边界与练习

这是最小教学版本，刻意没有数据筛选、超大规模/分布式训练（center 没有 all-reduce）、KoLeo、register tokens、FlashAttention 或真实 DINOv2 权重复现。`lesson*.py` 都故意保留 `NotImplementedError`；先独立完成，再对照参考答案。参考实现会显式拒绝无效 mask ratio、空 masked patch、错误 crop/mask 长度或不匹配的 token shape。
