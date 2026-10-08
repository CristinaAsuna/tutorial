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


## 从练习到整模型的学习路径

先完成共享基础练习并安装 `edu_core`，后续直接复用 patch/attention，不需要每篇重写。每个 lesson 的文件顶部包含目标、前置、张量形状、手算示例、常见错和接入位置。按编号补 TODO，先运行局部验收；卡住时按需展开 [HINTS.md](HINTS.md) 的三级提示。

默认 demo 验参考模型，默认关卡检查验自己的代码。以下命令从本目录运行（仓库使用的 Python 必须装有 torch 和 edu_core）：

```bash
python3 -m pip install -e ../../general_utils/edu_core
python3 check_lessons.py --lesson 1
python3 check_lessons.py --lesson 1 --implementation reference
python3 run_flamingo_demo.py --implementation reference
python3 run_flamingo_demo.py --implementation practice
python3 check_wiring.py
```

`check_lessons.py` 比较手算数值、padding/因果语义或共享权重下的参考输出；`--lesson` 只测该关（省略时按顺序检查全部关卡），不调用整篇 practice 模型。后续关的组合函数仍需要其声明的前置关完成。`check_wiring.py` 是维护者的接线检查：测试中注入 oracle 替身证明 demo 确实经过所有练习入口；它不补写学生答案。练习 demo 一旦遇到未完成 TODO 就在该函数抛 `NotImplementedError`，不会自动调用参考答案。

|关卡|实际学习内容|整模型接入|
|---|---|---|
|1|逐图固定R个latent|practice resampler.forward|
|2|sentinel累计、all-seen mask、padding|practice build_media_attention_mask|
|3|零初始化cross-attn与FFN门|practice decoder的第2/4层连接器|
|4|因果LM遍历和连接器插层、一次更新|practice decoder.forward及demo training step|

[practice_flamingo.py](practice_flamingo.py) 复用 mock vision、embedding与语言block初始化以及模型最外层loss/generate，替换Resampler、media mask、connector和decoder遍历。生成会反复经过学生forward；答案监督的next-token shift保留在外围以减少重复劳动。

## 论文核对与复现递进

依据 [Flamingo 论文 §2.1–2.3 与附录 A.1](https://arxiv.org/html/2204.14198v2#S2)。原论文主模型每个文本位置直接 cross-attend **紧邻其前的一幅图像**（immediate previous image）；更早图像信息可以通过语言self-attention间接传播。本 toy 明确采用 **all-seen** 教学策略：第二个sentinel后可读所有已经出现的图像，这是不同的mask契约。第2关完成后可自行增加immediate模式并相应修改验收，当前默认行为保持all-seen。

本 toy Resampler 是一次patch cross-attention加latent self-attention的简化版本。论文附录的多层Resampler会把视觉特征与latent都作为KV，并包含视频时间位置处理（论文不使用显式空间网格位置编码）；真实复现需恢复这些细节。toy connector放在指定冻结LM block之后，负责展示插层与零门控梯度；这不是论文checkpoint兼容结构。[OpenFlamingo源码](https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py) 可作公开实现补充，它是独立开源复现项目，并非原作者官方权重实现。

默认只监督答案的SFT loss也不同于论文混合网页数据的语言建模训练。recipe记录真实骨干、数据与训练条件；CPU验收不等于复现few-shot benchmark。
