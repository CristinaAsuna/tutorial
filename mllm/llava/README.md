# 从零教学实现 LLaVA-1.5（CPU Toy）

> 共享基础包：先执行 `python3 -m pip install -e "../../general_utils/edu_core[dev]"`。本文件描述 CPU toy；真实权重、数据、分布式配置与复现边界在 `recipe/`。

这个目录用一个可在 CPU 上跑的小模型讲清 LLaVA-1.5 的连接方式：冻结视觉编码器，将图像 patch 表征映射到 decoder LLM 的 token-embedding 空间，然后把 `<image>` 在文本序列中展开为一段连续视觉 token。它不是 CLIP/LLaMA 权重加载器，也不追求真实 benchmark 效果。

## 文件与学习顺序

```text
llava/
├── reference_llava.py              # 完整可运行参考解
├── conversation.py                 # prompt 与 assistant-only SFT labels
├── lesson1_vision_and_projector.py # TODO：视觉塔和两层 projector
├── lesson2_image_token_packing.py  # TODO：<image> 展开和 batch padding
├── lesson3_sft_loss.py             # TODO：SFT loss mask
├── lesson4_training_and_generation.py # TODO：两阶段训练和生成
└── run_llava_demo.py               # 只测试参考解
```

先运行参考 demo 理解验收目标，再按四个 lesson 独立实现。practice demo 实际依赖你的填答。

## LLaVA 与 BLIP-2 的关键不同

BLIP-2 在冻结视觉塔和 LLM 之间放入 Q-Former：少量可学习 query 从图像提取固定数量的视觉摘要，再投影成 LLM 的前缀。LLaVA-1.5 则直接使用视觉编码器的 patch tokens；经 MLP projector 后，`<image>` 的一个占位位置变成一串视觉 embedding。因此，LLM 的自回归注意力可以像读一段前缀 token 一样读图。

## 数据流和张量维度

本 toy 配置为 `B=2`、`H=W=8`、`patch=4`、`N_patch=4`、`D_vision=24`、`D_llm=32`。

|阶段|张量|形状|说明|
|---|---|---|---|
|视觉塔|`vision_encoder(images)`|`(B, 1+N_patch, D_vision)`|第一个是 CLS|
|丢 CLS|`features[:, 1:, :]`|`(B, N_patch, D_vision)`|LLaVA 路径只传 patch|
|projector|`Linear -> GELU -> Linear`|`(B, N_patch, D_llm)`|`24 -> 32 -> 32`|
|文本输入|`input_ids`|`(B, L_text)`|唯一的 `<image>` 是 `-200` sentinel|
|展开后|`inputs_embeds`|`(B, L_text-1+N_patch, D_llm)`|一个 sentinel 换成四个视觉 token|
|解码器输出|`logits`|`(B, L_expanded, V)`|因果语言模型 logits|

`IMAGE_TOKEN_INDEX=-200` 不是词表 ID，绝不能送进 `Embedding`。packing 会先定位它，再将它替换为 projector 输出；每条样本必须恰好有一个 sentinel 和一张图。batch 内文本原始长度可以不同，参考解一律**右侧 padding**，并同步构造 embedding、attention mask 和 labels。

## SFT 标签和 loss

`conversation.build_sft_example` 形成：

```text
BOS system tokens <image> user tokens assistant answer tokens EOS
```

只有 `assistant answer tokens + EOS` 是标签本身。system、user、`<image>`、展开出的所有视觉 patch、以及右侧 padding 的 label 都是 `-100` (`IGNORE_INDEX`)；交叉熵会忽略它们。语言模型用标准 next-token shift：第 `t` 个 logits 预测位置 `t+1` 的 label。

## 两阶段训练

1. `pretrain_projector`：Vision Encoder 和 LLM 的 `requires_grad=False`，只训练两层 projector，让视觉 token 对齐语言空间。
2. `instruction_tuning`：视觉塔仍冻结；projector 和 toy LLM 都可训练，以视觉指令对话的 assistant 回复为监督。

真实 LLaVA 的第二阶段常全参微调或 LoRA/QLoRA 微调 LLM；本教程用小 LLM 的全参路径，使梯度流动一目了然。无论调用 `model.train()` 与否，冻结视觉塔都被保持在 `eval()`，并在 `no_grad()` 下计算。

## 运行

安装 CPU 版 PyTorch 后：

```bash
cd /Users/max/codebase/scratch/tutorial/mllm/llava
python3 run_llava_demo.py
python3 -m compileall -q .
```

demo 检查 CLS 丢弃、视觉 token 展开、SFT/padding mask、两个训练阶段的梯度边界、一次参数更新，以及固定 eval 模式下的可复现贪心生成。

## 教学范围

刻意省略真实 CLIP/LLaMA、LoRA/QLoRA、多图和视频、任意分辨率、图像分块、KV cache、分布式训练及生产 tokenizer。当前接口的明确约束是：每条样本恰好一张图和一个 `<image>` sentinel。

## 动手学习闭环

基础 patchify/attention/EMA 先在共享基础课程练一次，本篇复用 `edu_core` 或冻结 toy 外围，重点实现论文机制。每个 lesson 已提供中文目标、前置、形状、小例子、编号 TODO 与常见错误；需要时逐级展开 [HINTS.md](HINTS.md)。

```bash
python3 check_lessons.py --lesson 1                 # 默认 practice，只检查这一关
python3 check_lessons.py --lesson 1 --implementation reference
python3 run_llava_demo.py --implementation reference          # 默认参考答案
python3 run_llava_demo.py --implementation practice           # 完成所有关后验证自己的闭环
python3 test_practice_wiring.py
```

四关依次执行 `--lesson 1` 至 `--lesson 4`。未完成时退出码 2 并指出论文、关卡和函数；不会自动回退参考答案。局部检查包含数值、标签或梯度语义。`practice_llava.py` 通过覆写关键方法/注入模块接入练习，复用的只有 mock 专家和外围冻结、损失 shift、贪心循环；训练步骤调用学生第四关。

`pack_one_image` 的完整接口包含文本 embedding、attention_mask 和 labels，返回三元组；`assistant_only_labels` 优先接显式 `assistant_mask`，兼容旧 `assistant_start`。单图负 sentinel 不进入 embedding。课程只支持每样本一幅固定尺寸图像，四个 patch，无真实 CLIP/Vicuna、tokenizer、多轮数据、AnyRes、权重兼容或真实 benchmark 复现。两层 GELU projector 对应 LLaVA-1.5；不能把它误认为最初版本的单线性 projector。

论文依据：[Visual Instruction Tuning 官方项目](https://llava-vl.github.io/)、[LLaVA-1.5 Improved Baselines 论文](https://arxiv.org/abs/2310.03744)、[官方代码](https://github.com/haotian-liu/LLaVA)。原始两阶段训练与本课冻结边界对应；所有 toy 随机 token 的 loss/生成只证明机制运行。

复杂批量拼接已提供验证、循环、去 padding 和右补齐骨架。LLaVA 第二关只需完成单样本替换与监督展开两个 helper；InstructBLIP 第三关拆为投影、单样本拼接、标签三个 helper。局部检查会独立检查每个 helper 并同时报告未完成子任务。
