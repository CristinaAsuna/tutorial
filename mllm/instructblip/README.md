# InstructBLIP：让 Q-Former 按指令读取图像（CPU Toy）

InstructBLIP 是对 BLIP-2 的关键扩展：BLIP-2 Stage 2 的 Q-Former 只从图像提取固定 visual queries；InstructBLIP 先把 **learnable queries 与 instruction tokens** 放进同一个 Q-Former self-attention 序列，使指令改变 query 从图像读取什么，随后再将 query 投影为冻结 LLM 的 visual soft prefix。

```text
image -> frozen vision encoder -> image patches (B,N,Dv)
instruction ids -> Q-Former text embedding -----------+
learnable queries ------------------------------------+-> self-attention
                                                          -> query-only visual cross-attention
                                                          -> instruction-aware queries (B,M,Dq)
                                                          -> projection + frozen LLM prompt -> answer loss
```

## 两套 tokenizer，两个职责

- `instruction_ids` 属于 Q-Former 的文本词表，负责调制视觉 query；它与 query 一起 self-attend，但 instruction token **不直接** cross-attend 到 image patches。
- `llm_prompt_ids` 属于 LLM 的词表，负责告诉冻结 LLM 如何把 visual prefix 转成答案。

真实模型常使用不同 tokenizer，因此本教程刻意使用两个输入，避免把它们误当成同一个 ID 空间。训练时只有 answer token 参与 next-token loss；视觉 prefix、prompt 与 padding 都是 `-100`。

## 文件与运行

```text
instructblip/
├── reference_instructblip.py       # 可运行完整答案
├── lesson1 ... lesson4             # TODO 练习
├── run_instructblip_demo.py        # instruction effect / gradients / generation
└── recipe/                         # 真实训练条件合同
```

```bash
cd /Users/max/codebase/scratch/tutorial/mllm/instructblip
PYTHONPATH=../../general_utils/edu_core /Users/max/codebase/.ml/.venv/bin/python run_instructblip_demo.py
PYTHONPATH=../../general_utils/edu_core /Users/max/codebase/.ml/.venv/bin/python recipe/validate_config.py
```

## 教学边界

初版使用 mock vision/LLM、固定小图像与 toy token IDs；不加载 EVA-CLIP、Flan-T5/Vicuna，不实现 26 数据集混合、真实 template、LoRA、分布式训练或 benchmark 复现。它实现的是 InstructBLIP 相对 BLIP-2 最重要的 instruction-aware Q-Former 机制与训练边界。
