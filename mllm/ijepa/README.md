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

## 从练习到 CPU 闭环

先完成 [基础练习](../foundations/README.md)，再复用 patch/token、attention、EMA；此目录只练论文独有的路由和目标。每个 lesson 提供形状、手算例子、编号步骤与常见错误；卡住时逐层展开 [HINTS.md](HINTS.md)。

```bash
# 在当前目录执行；使用已安装 torch 的 Python。
python3 check_lessons.py --lesson 1
python3 check_lessons.py --lesson 1 --implementation reference
python3 run_ijepa_demo.py --implementation reference
python3 run_ijepa_demo.py --implementation practice
```

`--lesson N` 支持 1–4，只检验指定关卡；省略时遍历全部关卡。指定单关时允许后面的 TODO 未完成。`practice_ijepa.py` 将实际学生函数装入模型；未完成会显示准确的文件和函数，不会替换为答案。练习完成后，practice demo 与 reference demo 使用相同形状、梯度、参数更新与 teacher 验收。参考解通过只说明基础环境可运行。

## 核对来源与 toy 边界

已核对 [官方 I-JEPA train.py](https://github.com/facebookresearch/ijepa/blob/main/src/train.py) 的 `forward_target`：teacher 完整图像编码、feature norm 后选择预测 mask；context 先选可见 tokens，优化后 EMA。本仓库已修正 teacher 曾在编码前只选 target 的错误。`check_lessons.py --lesson 4 --implementation reference` 用 hook 确认 teacher Transformer 收到全部 16 个 tokens，并验证 target 等于完整编码后 gather。

本 toy 采用固定网格、共享 batch mask、不重叠等大矩形、context 精确补集、合并 target slots 及 learned positions；这些是教学合同，未实现论文多尺度 masks、每块独立预测和真实训练评测。

练习 API：`practice_ijepa.build_toy_ijepa()` 返回与 demo 同尺寸的 `IJEPA` 模型。模型 forward 返回 dictionary，含 `loss`、`predictions` 和 `targets`；`loss` 使用第 4 关学生回归函数。
