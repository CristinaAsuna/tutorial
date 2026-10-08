"""关卡 1：可学习 query 与 instruction 组成联合序列。
目标：指令能影响视觉查询；前置：Embedding、expand、cat、有效 token mask。
形状：query (1,M,D)，instruction (B,L,D) -> tokens (B,M+L,D), valid (B,M+L)。
手算：M=2,L=3，指令 mask=[1,1,0] -> valid=[1,1,1,1,0]。
常见错：query 放在末尾；把 Q-Former 指令 id 当成 LLM prompt id。
接入：PracticeQFormer.forward，位置编码已由外围加入 instruction_embeds。
"""
import torch

def instruction_aware_queries(query_tokens, instruction_embeds, instruction_attention_mask=None):
    # TODO 1.1：检查 (1,M,D)/(B,L,D)，将 query expand 到 B，不复制独立参数。
    # TODO 1.2：cat([queries,instruction],dim=1)。
    # TODO 1.3：mask 默认全有效；给 query prepend M 个 True，返回 tokens,valid。
    raise NotImplementedError("InstructBLIP lesson 1: instruction_aware_queries 尚未完成；见 HINTS.md")
