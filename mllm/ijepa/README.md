# I-JEPA：从图像上下文预测隐空间目标

这是原始 I-JEPA 的 CPU toy 教学实现：student 只看 context patches，EMA teacher 看完整图像；predictor 利用 target 的位置和 mask token 预测 teacher 的 target latent。训练没有标签、没有 negative pairs、没有 pixel decoder。

```text
BCHW image -> patch tokens (B, 16, D)
target blocks -> hidden target indices (B, 8, D)
context complement -> student context (B, 8, D)
context + target slots -> predictor -> (B, 8, D)
EMA teacher full image -> target latents -> Smooth-L1 loss
```

## 与已有教程的区别

- MAE：预测被遮蔽 patch 的归一化像素。
- DINO/iBOT：用 teacher 的 soft distribution 做跨视图/patch 蒸馏。
- I-JEPA：直接回归 teacher representation，因此不需要 pixels、labels、prototypes 或 negatives。

## 文件与运行

- `reference_ijepa.py`：可运行参考答案。
- `lesson1` 到 `lesson4`：故意保留 TODO 的练习关卡。
- `run_ijepa_demo.py`：固定种子的 CPU 训练、EMA 和确定性 eval 验证。
- `recipe/`：论文训练条件合同，而非已验证的大规模复现。

```bash
PYTHONPATH=../../general_utils/edu_core /Users/max/codebase/.ml/.venv/bin/python run_ijepa_demo.py
PYTHONPATH=../../general_utils/edu_core /Users/max/codebase/.ml/.venv/bin/python recipe/validate_config.py
```

限制：初版使用固定 `32x32` 图像和一个 batch 共享的矩形 mask，目的是让 token 的选择与预测关系可审计；真实 I-JEPA 使用多尺度随机 context/target blocks 和大规模 ImageNet 训练。
