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

## 动手学习闭环

基础 patchify/attention/EMA 先在共享基础课程练一次，本篇复用 `edu_core` 或冻结 toy 外围，重点实现论文机制。每个 lesson 已提供中文目标、前置、形状、小例子、编号 TODO 与常见错误；需要时逐级展开 [HINTS.md](HINTS.md)。

```bash
python3 check_lessons.py --lesson 1                 # 默认 practice，只检查这一关
python3 check_lessons.py --lesson 1 --implementation reference
python3 run_instructblip_demo.py --implementation reference          # 默认参考答案
python3 run_instructblip_demo.py --implementation practice           # 完成所有关后验证自己的闭环
python3 test_practice_wiring.py
```

四关依次执行 `--lesson 1` 至 `--lesson 4`。未完成时退出码 2 并指出论文、关卡和函数；不会自动回退参考答案。局部检查包含数值、标签或梯度语义。`practice_instructblip.py` 通过覆写关键方法/注入模块接入练习，复用的只有 mock 专家和外围冻结、损失 shift、贪心循环；生成也调用学生 prefix packing（空 answer），训练步骤调用学生第四关。

两套 token id 显式分开：`instruction_ids` 使用 Q-Former 的 vocab，`llm_prompt_ids/answer_ids` 使用 LLM vocab。这里手工构造 ids，并未实现实际 tokenizer。query 与 instruction 联合 self-attend 后，仅 query cross-attend vision；LLM 接收投影后的 query prefix、有效 prompt、有效 answer，只有 answer 被监督。练习第三关的完整接口返回 embeds/mask/labels，保留旧二参数纯 embedding 拼接入口。

论文依据：[InstructBLIP 论文](https://arxiv.org/abs/2305.06500)、[Salesforce 官方项目与模型](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip)。本课演示 instruction-aware 提取和冻结专家，使用随机 tiny 视觉塔/Q-Former/decoder；没有真实 BLIP-2 预训练权重、BERT query/text 分支结构、26 数据集处理、13 held-out 评测或真实 tokenizer。CPU 通过不代表论文指标复现，真实条件仍见 recipe。

复杂批量拼接已提供验证、循环、去 padding 和右补齐骨架。LLaVA 第二关只需完成单样本替换与监督展开两个 helper；InstructBLIP 第三关拆为投影、单样本拼接、标签三个 helper。局部检查会独立检查每个 helper 并同时报告未完成子任务。
