# Flamingo 真实训练 recipe 合同

此目录说明从 CPU toy 走向论文级 Flamingo 需要的外部条件：冻结的视觉/语言预训练权重、M3W 式交错网页图文、普通图文与视频文本数据混合、长上下文训练、bf16 分布式运行、checkpoint/resume，以及 few-shot benchmark 协议。

顶层代码只验证结构与梯度边界；它不含数据许可、视频解码、真实预训练权重或论文评测，不能据此宣称复现 Flamingo 分数。运行 `python validate_config.py` 仅检查配置合同。
