# 从零理解 V-JEPA：视频 latent prediction

> 共享基础包：`python -m pip install -e "../../general_utils/edu_core[dev]"`。顶层实现是 CPU toy 教学；`recipe/` 只记录真实训练的前提。

V-JEPA（Video Joint-Embedding Predictive Architecture）不重建被遮住的 RGB 像素。它让 student 的 **context encoder** 只看可见 tubelet，借助 target 的时空位置预测 EMA teacher 对被遮住 tubelet 给出的 latent representation。

```text
video [B,C,T,H,W]
  -> TubeletEmbed -> N=(T/t)*(H/p)*(W/p) video tokens
  -> target cuboids / context complement
     context tokens -> context VideoViT ---------+
     target positions -> mask tokens ------------+-> predictor -> [B,M,D] prediction
     complete video -> frozen EMA target VideoViT -> [B,M,D] target
  -> Smooth-L1(prediction, stop_gradient(target))
```

## 与相邻教程的关系

| 方法 | 预测什么 | target 来源 | 是否重建像素 |
| --- | --- | --- | --- |
| MAE | masked patch pixels | 原视频/图像 | 是 |
| DINO/iBOT | prototype distribution | EMA teacher | 否 |
| I-JEPA | image patch latent | EMA teacher | 否 |
| **V-JEPA** | video tubelet latent | EMA teacher | 否 |

`reference_vjepa.py` 采用固定的 `(T,H,W)` 网格，便于教学时精确检查 index。`sample_spatiotemporal_masks` 保证 target cuboids 不重叠，且 `context_mask == ~target_mask`；因此 student 不会把任何 target tubelet 作为输入。Teacher 读取完整视频是预测目标的定义，不是给 student 的泄漏通道。

## 文件与运行

```text
vjepa/
├── reference_vjepa.py       # 完整参考解
├── lesson1_tubelet_embed.py # 3D token 化 TODO
├── lesson2_spatiotemporal_masks.py
├── lesson3_latent_predictor.py
├── lesson4_ema_training.py
├── run_vjepa_demo.py
└── recipe/                  # 真实训练合同，不是论文复现声明
```

```bash
PYTHONPATH=../../general_utils/edu_core \
  /Users/max/codebase/.ml/.venv/bin/python run_vjepa_demo.py
```

真实 V-JEPA 还需要视频数据解码、增广、长 schedule、分布式训练和标准下游评测。这里刻意不实现 V-JEPA 2 的 dense/deep supervision 或 action-conditioned world model，以保持原始 V-JEPA 的教学边界。

## 从练习到 CPU 闭环

先完成 [基础练习](../foundations/README.md)，再复用 patch/token、attention、EMA；此目录只练论文独有的路由和目标。每个 lesson 提供形状、手算例子、编号步骤与常见错误；卡住时逐层展开 [HINTS.md](HINTS.md)。

```bash
# 在当前目录执行；使用已安装 torch 的 Python。
python3 check_lessons.py --lesson 1
python3 check_lessons.py --lesson 1 --implementation reference
python3 run_vjepa_demo.py --implementation reference
python3 run_vjepa_demo.py --implementation practice
```

`--lesson N` 支持 1–4，只检验指定关卡；省略时遍历全部关卡。指定单关时允许后面的 TODO 未完成。`practice_vjepa.py` 将实际学生函数装入模型；未完成会显示准确的文件和函数，不会替换为答案。练习完成后，practice demo 与 reference demo 使用相同形状、梯度、参数更新与 teacher 验收。参考解通过只说明基础环境可运行。

## 核对来源与 toy 边界

已核对 [官方 V-JEPA train.py](https://github.com/facebookresearch/jepa/blob/main/app/vjepa/train.py)：teacher 全视频无梯度编码后取目标区域，context encoder/predictor 使用 masks，optimizer 后 EMA。

本 toy 使用 Smooth-L1 便于延续 I-JEPA；官方 V-JEPA 使用可配置 `abs(z-h)**loss_exp / loss_exp`，因此本例不是论文损失的完整复刻。固定网格、learned positions、等大不重叠 cuboids、精确 complement、合并 target slots、CPU 随机数据也都是教学简化。实践检查会验证原始 tubelet 展平与 Conv3d 投影一致。

练习 API：`practice_vjepa.build_toy_vjepa()` 返回与 demo 同尺寸的 `VJEPA` 模型。模型 forward 返回 dictionary，含 `loss`、`predictions` 和 `targets`；`loss` 使用第 4 关学生回归函数。
