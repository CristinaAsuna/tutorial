# Modern MLLM / vision-learning tutorials

每篇目录都有两条明确分开的轨道：顶层 `reference_*.py` 和 `lesson*.py` 是小模型、CPU toy 的机制教学；`recipe/` 是真实数据、外部预训练权重和分布式训练所需的配方骨架。后者不会因为配置文件存在就声称已复现论文指标。

当前教程包括：

- `mae/`：masked-pixel reconstruction；`dino/`：DINO + iBOT 的无标签 self-distillation；
- `ijepa/`：图像 context 到 EMA target latent 的预测；`vjepa/`：原始 V-JEPA 的视频 tubelet latent prediction；
- `blip2/`、`instructblip/`、`llava/`、`flamingo/`：视觉 token 接入语言模型的多模态训练链路；InstructBLIP 展示 instruction-aware Q-Former，Flamingo 展示交错多图文本与 gated cross-attention。

先安装共享基础包：

```bash
python3 -m pip install -e "../general_utils/edu_core[dev]"
```

`general_utils/utils` 继续服务扩散/生成模型。MLLM 使用 `general_utils/edu_core` 提供的 `edu_core` token 语义；其中 `TubeletEmbed`、3D position interpolation 与 EMA schedule 由 I-JEPA/V-JEPA 等教程共享，避免重复实现且不把空间 GroupNorm attention 或 `[-1,1]` 图像约定混入 token 序列训练。
