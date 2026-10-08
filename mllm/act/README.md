# ACT：Action Chunking with Transformers（CPU Toy）

ACT 是用于精细双臂 manipulation 的视觉模仿学习策略。给定当前多相机图像和机器人关节位置，它不只预测下一个关节 target，而是预测未来 `K` 步的 **absolute joint-position action chunk**；每个时刻都重新预测一个 chunk，并把重叠 chunk 中“针对同一当前时刻”的 action 做 temporal ensemble。

```text
current RGB views + follower qpos
  -> visual tokens + qpos token + CVAE style token z
  -> Transformer encoder memory
  -> K learned action queries + Transformer decoder
  -> future absolute joint chunk: (B, K, A)

training: q(z | qpos, demonstrated future actions) -> L1(valid actions) + beta * KL
inference: z = 0 -> predicted chunks -> temporal ensemble -> one executed action
```

## 论文机制与教学实现

- **Action chunking**：学习 `π(a[t:t+K] | o[t])`，减少长轨迹的有效决策 horizon；不是一次预测整条 episode。
- **Temporal ensemble**：每一步重新观察并预测，融合重叠 chunk 中对当前时刻的多个预测，避免每 `K` 步才更新造成的突变。
- **CVAE**：训练时 style encoder 由当前 proprioception 与 demonstration action chunk 推断 `z`；推理时 encoder 被丢弃并固定 `z=0`。这用于吸收人类 demonstration 的噪声与多模态性。

`reference_act.py` 使用小型 Conv + Transformer encoder/decoder，保留论文的数据流而不是 ALOHA 的 ResNet-18、4 路 480×640 相机与 14-DoF 双臂硬件。

## 文件与运行

```text
act/
├── reference_act.py            # 完整参考答案
├── lesson1 ... lesson4         # action chunk / CVAE / policy / ensemble TODO
├── run_act_demo.py             # CPU contract checks
└── recipe/                     # 真机数据与控制接口契约
```

```bash
cd /Users/max/codebase/scratch/tutorial/mllm/act
PYTHONPATH=../../general_utils/edu_core /Users/max/codebase/.ml/.venv/bin/python run_act_demo.py
PYTHONPATH=../../general_utils/edu_core /Users/max/codebase/.ml/.venv/bin/python recipe/validate_config.py
```

## 边界

本教程不实现 teleoperation、ROS、Dynamixel PID、相机时间同步、真机 safety limits 或 benchmark rollout。它只验证 ACT policy 的张量语义、valid action mask、CVAE 训练/确定性推理与 temporal ensemble；真实硬件路径必须先完成 recipe 中的数据、标定与安全接口。
