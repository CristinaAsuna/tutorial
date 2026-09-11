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

先读并运行参考解，再按四个 lesson 补全 TODO。lesson 故意抛出 `NotImplementedError`；端到端 demo 不依赖它们。

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
