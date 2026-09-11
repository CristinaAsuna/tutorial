# LLaVA-1.5 recipe track

真实 recipe 使用 checkpoint 对应的 CLIP vision tower feature layer、patch feature selection、`mlp2x_gelu` projector、Vicuna/LLaMA tokenizer 与 conversation template。第一阶段训练 projector alignment，第二阶段以 visual instruction SFT 训练；LLM 的 LoRA 或全参选择必须写入 checkpoint metadata。

真实 image processor、tokenizer、template、数据许可和 DeepSpeed/DDP 配置都属于 recipe 输入。顶层实现仍限制为单图、toy token IDs 和 mock backbone，不能被用于报告 LLaVA-1.5 指标。
