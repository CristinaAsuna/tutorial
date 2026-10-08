"""关卡 3：多相机 image tokens + qpos + z 作为 memory，K 个 action query 解码动作序列。"""


def decode_action_chunk(observation_memory, action_queries):
    # TODO: TransformerDecoder 用 action_queries cross-attend memory；输出 (B,K,A)。
    raise NotImplementedError("参考 ACTPolicy._decode")
