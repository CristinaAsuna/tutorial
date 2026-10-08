"""关卡 3：零门控的 cross-attention 和 FFN。
目标：新视觉连接器在 step0 是恒等映射，逐渐开放视觉支路。
前置：LayerNorm、residual、tanh；复用 edu_core.MultiHeadAttention，不再手写 QKV。
形状：x(B,L,D), memory(B,M*R,D), allow(B,L,M*R) -> (B,L,D)。
手算：gate=0 时 tanh=0，所以输出严格等于 x；gate增大后视觉变化可以影响输出。
常见错：gate初始化为1；用sigmoid导致初始非恒等；只写attention门不写FFN门。
接入：practice decoder 在第2/4层后使用本类，不使用共享答案的 gated forward。
"""
import torch
from torch import nn
from edu_core.attention import MultiHeadAttention

class GatedCrossAttention(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=2.):
        super().__init__()
        self.norm1=nn.LayerNorm(dim)
        self.cross_attn=MultiHeadAttention(dim,heads)
        self.norm2=nn.LayerNorm(dim)
        self.mlp=nn.Sequential(nn.Linear(dim,int(dim*mlp_ratio)),nn.GELU(),nn.Dropout(0.),nn.Linear(int(dim*mlp_ratio),dim))
        self.attn_gate=nn.Parameter(torch.zeros(()))
        self.ff_gate=nn.Parameter(torch.zeros(()))
    def forward(self,x,context,*,context_padding_mask=None,attention_mask=None):
        # TODO 3.1：norm1(x)查context；传入两种mask，乘tanh(attn_gate)，加x。
        # TODO 3.2：基于更新后的x算mlp(norm2(x))，乘tanh(ff_gate)，加residual。
        raise NotImplementedError("Flamingo TODO 3.1-3.2: GatedCrossAttention.forward")
