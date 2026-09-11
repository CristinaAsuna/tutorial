# MAE recipe track

这是 ImageNet 风格 MAE 预训练的配置合同，不是顶层 CPU toy 的扩容版本。训练实现应使用 `224x224` random-resized crop、ViT-L/16、75% random masking、sin-cos position embedding、AdamW、warmup + cosine，并保存预训练 checkpoint、linear probe 与 fine-tune 的独立结果。

`config.json` 记录默认论文级目标；实际运行前必须提供有授权的 ImageNet 路径、DDP/AMP 环境和模型 adapter。验收不只看 reconstruction loss，还应包含固定 protocol 的 linear probe/fine-tune 指标。`validate_config.py` 仅验证配置和数据路径，不会假装启动训练。
