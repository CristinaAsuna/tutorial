"""关卡 2：Q-Former 中只有 learnable query 可 cross-attend 到 image patches。"""


def query_only_cross_attention(hidden_states, num_queries, image_features):
    # TODO: 取前 M 个 query 查视觉 token，再把 instruction token 原样拼回。
    raise NotImplementedError("参考 InstructionQFormerLayer.forward")
