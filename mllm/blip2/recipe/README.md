# BLIP-2 recipe track

真实 recipe 必须由 adapter 加载并冻结 EVA-CLIP/CLIP、BERT 初始化的 Q-Former 和 OPT 或 Flan-T5；stage 1 采用 ITC/ITM/ITG，ITC negatives 必须跨卡聚合，stage 2 使用投影后的 visual query prefix。图像归一化由视觉 checkpoint 决定，不能使用生成模型的 `[-1,1]` 默认值。

顶层 `MockVisionEncoder` 和 `MockLLM` 只用于解释张量流。真实运行前需接受模型权重许可、数据许可、tokenizer 版本和 prompt 格式，并在 checkpoint metadata 中持久化它们。
