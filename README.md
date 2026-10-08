# Generative AI & MLLM — From-Scratch Tutorials

这是一个以 PyTorch 为主的个人学习仓库：通过推导、小例子、分步练习和训练闭环学习论文算法，保留生成模型、视觉自监督、多模态与动作策略实现。

## Repository layout

| Directory | 内容 |
| --- | --- |
| [`mllm/`](mllm/) | MAE、DINO + iBOT、I-JEPA、原始 V-JEPA、BLIP-2、InstructBLIP、LLaVA、Flamingo，以及 ACT embodied policy 的机制教学实现、TODO 关卡与 paper-recipe 配置合同。|
| [`generative_model/`](generative_model/) | AE、KL-VAE、VQ-VAE、VQGAN、DDPM、Flow Matching、VP-SDE Score Matching、扩散 Transformer 与 latent-diffusion 实验。|
| [`general_utils/`](general_utils/) | `utils/`：生成模型的通用 2D/训练工具；`edu_core/`：token attention、ViT token、mask、EMA、batching 等跨教程基础组件。|

`mllm/` 顶层 reference/demo 是可验证的 CPU toy，用于理解论文数据流；每篇的 `recipe/` 记录真实数据、预训练权重、分布式训练和指标所需的条件。它们不是对论文结果的未经验证声明。

## Setup

在包含 PyTorch 的环境中安装 MLLM 共享基础包：

```bash
python3 -m pip install -e "general_utils/edu_core[dev]"
python3 -m pytest -q general_utils/edu_core/tests
```

例如运行 LLaVA 的 CPU demo：

```bash
cd mllm/llava
python3 run_llava_demo.py
```

生成模型代码从仓库根目录执行时，使用 `general_utils.utils` 导入其训练、数据、checkpoint 和空间 attention 工具。

## 开始动手

在 Codex 中选择本项目新建 session，输入论文名（例如 `I-JEPA`）即可按仓库约定创建或完善从零实现课程。默认采用中文细步骤、分层提示、CPU toy 与练习闭环；同名论文存在歧义时才确认版本。完整约定见 [论文名称入口工作流](docs/paper_workflow.md)。

先读 [课程地图](mllm/README.md)。基础不熟可以从 [foundations](mllm/foundations/README.md) 的 patchify、attention、mask 和 EMA 开始；后续论文直接复用 `edu_core`，关键论文机制自己填写。

从仓库根目录检查一关，然后在填完论文练习后运行自己的模型：

```bash
PYTHONPATH=general_utils/edu_core python3 mllm/foundations/check_lessons.py --lesson 1
PYTHONPATH=general_utils/edu_core python3 mllm/llava/check_lessons.py --lesson 1
PYTHONPATH=general_utils/edu_core python3 mllm/llava/run_llava_demo.py --implementation practice
```

每关先读 lesson 的推导、小例子和编号 TODO，卡住时再看对应 HINTS.md。默认参考 demo 可用来理解完整数据流；practice 模式才会运行你的填答，未完成会指出 TODO。

维护者入口：`python3 mllm/check_tutorials.py --implementation reference` 检查七篇参考课程和 demo；[项目记忆](docs/project_memory.md) 记录教学决策、验证状态与学习进度模板。
