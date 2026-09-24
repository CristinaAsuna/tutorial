"""关卡 1：把 learnable query 与 instruction token 拼入同一个 Q-Former 序列。"""


def instruction_aware_queries(query_tokens, instruction_embeds):
    # TODO: 扩展 query 到 batch，拼接 [queries, instruction]；保持 query 在前。
    raise NotImplementedError("参考 reference_instructblip.InstructionAwareQFormer")
