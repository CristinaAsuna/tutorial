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
