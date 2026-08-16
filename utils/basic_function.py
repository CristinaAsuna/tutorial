import torch
from torch import nn
from torch.nn import functional as F

class Residualblock(nn.Module):
    def __init__(self, inc, outc, dropout=0.0):
        super().__init__()

        self.norm1 = nn.GroupNorm(32, inc, eps=1e-6)
        self.norm2 = nn.GroupNorm(32, outc, eps=1e-6)

        self.conv1 = nn.Conv2d(inc, outc, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(outc, outc, kernel_size=3, padding=1)

        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Conv2d(inc, outc, kernel_size=1)
            if inc != outc else nn.Identity()
        )

    def forward(self, x):
        residual = self.residual(x)

        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(self.dropout(F.silu(self.norm2(x))))

        return x + residual