# InstructBLIP 真实训练 recipe 合同

论文级训练需要冻结的 EVA-CLIP 与 LLM 权重、覆盖 VQA/captioning/reasoning/classification 等任务的多数据集 instruction mixture、任务模板和各 tokenizer、bf16 分布式运行、checkpoint/resume，以及跨数据集的 zero-shot/few-shot 评测协议。

顶层 toy 只实现 instruction-aware Q-Former 如何改变 visual query，并验证冻结梯度边界；它不包含 26 个数据集、真实 tokenizer、许可、权重或论文分数。`python validate_config.py` 仅检查配置契约。
