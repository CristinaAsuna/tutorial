import torch
from torch import nn
from torch.nn import functional as F

class SpatialMHSA(nn.Module):
    def __init__(self, channels, nheads=8, dropout=0.0):
        super().__init__()
        assert channels % nheads == 0

        self.nheads = nheads
        self.head_dim = channels // nheads
        self.dropout = dropout

        self.norm = nn.GroupNorm(32, channels)
        self.to_qkv = nn.Linear(channels, channels * 3)
        self.to_out = nn.Linear(channels, channels)

    def forward(self, x):
        # x: (B, C, H, W)
        residual = x
        b, c, h, w = x.shape

        # 先归一化，再把空间位置展平为 token
        x = self.norm(x)
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        q = q.view(b, h * w, self.nheads, self.head_dim).transpose(1, 2)
        k = k.view(b, h * w, self.nheads, self.head_dim).transpose(1, 2)
        v = v.view(b, h * w, self.nheads, self.head_dim).transpose(1, 2)

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
        )

        x = x.transpose(1, 2).contiguous().view(b, h * w, c)
        x = self.to_out(x)

        x = x.transpose(1, 2).reshape(b, c, h, w)

        # 注意力模块通常应保留残差连接
        return x + residual