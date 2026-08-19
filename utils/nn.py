from torch import nn


def make_group_norm(channels, max_groups=32):
    """创建 group 数可整除通道数的 GroupNorm。"""
    groups = min(channels, max_groups)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels, eps=1e-6)
