# Flamingo：交错多图文本的视觉语言模型（CPU Toy）

本教程从零实现 Flamingo 最有辨识度的数据流：冻结视觉 encoder 与 decoder LLM；每张图先经 **Perceiver Resampler** 压缩为固定数目的 visual latents；冻结 LM 的若干层之间插入可训练的 **gated cross-attention**，按图像在 prompt 中出现的顺序读取视觉记忆。

```text
images (B,M,C,H,W) -> frozen vision patches (B,M,N,Dv)
                       -> Perceiver Resampler -> (B,M,R,Dlm)

text: BOS ... <image> text ... <image> answer
                       -> frozen causal LM layers
                          + gated cross-attention every K layers
                          + media-causal mask: only images on the left are visible
                       -> next-token loss (answer only in this SFT toy)
```

## 为什么不是把 image token 插入文本序列？

LLaVA 将每个 `<image>` 展开成一长段 projected patch embeddings。Flamingo 不这样做：每幅图的可变 patch 数先被 Resampler 压缩为固定 `R` 个视觉记忆 token，文本长度不随 patch 数增长；语言 token 在 LM 内部的 gated cross-attention 中查询这些记忆。

它与 BLIP-2 也都使用 learnable queries/latents 形成视觉 bottleneck，但职责不同：BLIP-2 Q-Former 还承担明确的图文对齐训练并把 query 输出作为 LLM soft prompt；Flamingo Resampler 是视觉压缩器，语言条件融合由多层 gated cross-attention 完成。

## 文件与运行

```text
flamingo/
├── reference_flamingo.py            # 完整答案
├── conversation.py                  # 交错 prompt 和 answer-only labels
├── lesson1 ... lesson4              # 带 TODO 的递进练习
├── run_flamingo_demo.py             # CPU 端到端验证
└── recipe/                          # 真实训练条件合同
```

```bash
cd /Users/max/codebase/scratch/tutorial/mllm/flamingo
PYTHONPATH=../../general_utils/edu_core /Users/max/codebase/.ml/.venv/bin/python run_flamingo_demo.py
PYTHONPATH=../../general_utils/edu_core /Users/max/codebase/.ml/.venv/bin/python recipe/validate_config.py
```

## 教学边界

初版只支持每个样本相同数目的静态图像与严格 right-padding；每张图恰好对应一个 `<image>` sentinel。它不实现论文的真实 NFNet/Chinchilla、视频帧、M3W 数据管线、KV cache 或 benchmark 复现。`recipe/` 记录这些真实训练所需的契约，不是可直接报告论文指标的训练脚本。
