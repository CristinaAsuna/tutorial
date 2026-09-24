# Generative AI & MLLM — From-Scratch Tutorials

这是一个以 PyTorch 为主的个人学习仓库：既保留生成模型的数学与训练实现，也新增现代视觉自监督与多模态大模型的逐步教学代码。

## Repository layout

| Directory | 内容 |
| --- | --- |
| [`mllm/`](mllm/) | MAE、DINO + iBOT、I-JEPA、原始 V-JEPA、BLIP-2、InstructBLIP、LLaVA、Flamingo 的机制教学实现、TODO 关卡与 paper-recipe 配置合同。|
| [`generative_model/`](generative_model/) | AE、KL-VAE、VQ-VAE、VQGAN、DDPM、Flow Matching、VP-SDE Score Matching、扩散 Transformer 与 latent-diffusion 实验。|
| [`general_utils/`](general_utils/) | `utils/`：生成模型的通用 2D/训练工具；`edu_core/`：token attention、ViT token、mask、EMA、batching 等跨教程基础组件。|

`mllm/` 顶层 reference/demo 是可验证的 CPU toy，用于理解论文数据流；每篇的 `recipe/` 记录真实数据、预训练权重、分布式训练和指标所需的条件。它们不是对论文结果的未经验证声明。

## Setup

在包含 PyTorch 的环境中安装 MLLM 共享基础包：

```bash
python -m pip install -e "general_utils/edu_core[dev]"
python -m pytest -q general_utils/edu_core/tests
```

例如运行 LLaVA 的 CPU demo：

```bash
cd mllm/llava
python run_llava_demo.py
```

生成模型代码从仓库根目录执行时，使用 `general_utils.utils` 导入其训练、数据、checkpoint 和空间 attention 工具。
