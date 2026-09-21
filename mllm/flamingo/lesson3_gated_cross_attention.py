"""关卡 3：在冻结 LM 层之间加入零初始化 tanh gate 的 cross-attention。"""
from torch import nn


class GatedCrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        # TODO: LayerNorm -> cross-attn -> tanh(alpha) residual；再加入 gated FFN。
        raise NotImplementedError("参考 edu_core.attention.GatedCrossAttentionBlock")
