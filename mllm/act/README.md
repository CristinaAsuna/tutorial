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


## 从练习到整模型的学习路径

先完成共享基础练习并安装 `edu_core`，后续直接复用 patch/attention，不需要每篇重写。每个 lesson 的文件顶部包含目标、前置、张量形状、手算示例、常见错和接入位置。按编号补 TODO，先运行局部验收；卡住时按需展开 [HINTS.md](HINTS.md) 的三级提示。

默认 demo 验参考模型，默认关卡检查验自己的代码。以下命令从本目录运行（仓库使用的 Python 必须装有 torch 和 edu_core）：

```bash
python3 -m pip install -e ../../general_utils/edu_core
python3 check_lessons.py --lesson 1
python3 check_lessons.py --lesson 1 --implementation reference
python3 run_act_demo.py --implementation reference
python3 run_act_demo.py --implementation practice
python3 check_wiring.py
```

`check_lessons.py` 比较手算数值、padding/因果语义或共享权重下的参考输出；`--lesson` 只测该关（省略时按顺序检查全部关卡），不调用整篇 practice 模型。后续关的组合函数仍需要其声明的前置关完成。`check_wiring.py` 是维护者的接线检查：测试中注入 oracle 替身证明 demo 确实经过所有练习入口；它不补写学生答案。练习 demo 一旦遇到未完成 TODO 就在该函数抛 `NotImplementedError`，不会自动调用参考答案。

|关卡|实际学习内容|整模型接入|
|---|---|---|
|1|未来索引、末尾valid mask|demo数据准备|
|2|posterior token packing、重参数化、valid L1与KL|practice style encoder和forward|
|3|camera/spatial位置、qpos/z memory、action queries|practice _encode_observation/_decode|
|4|当前时刻offset、历史容量、指数融合|practice TemporalEnsembler.add|
|5|训练posterior与推理z=0、一次更新|practice forward及demo training_step|

[practice_act.py](practice_act.py) 借用参考类的参数初始化与 `predict_chunk` 外围，覆盖全部论文机制；第2关拥有独立的 StyleEncoder 模块骨架，第5关只负责把前序实现串起来。改变练习函数后，整模型运行会立即使用该变化。

## 论文核对与复现递进

依据 [ACT 论文](https://arxiv.org/abs/2304.13705) 和 [作者官方实现](https://github.com/tonyzhaozh/act)。先读论文的 ACT/CVAE/temporal ensemble 部分，再对照这里的三个机制；官方代码的 temporal aggregation 对较早预测使用更大指数权重，当前 toy 保留此方向。当前动作是 absolute joint-position target，而非 delta。

toy 用Conv patch表示和可学习位置替代真实视觉骨干。CPU闭环验张量与梯度机制，并不证明机器人任务成功率。后续复现要按 recipe 接入真实示范轨迹、相机/关节同步与ResNet策略骨干，再补训练/rollout指标。
