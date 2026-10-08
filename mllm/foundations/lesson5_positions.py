"""基础 5：二维学习位置网格插值。

前置：基础 1；interpolate 操作的是空间网格，不能把 CLS 混进去。
形状：(1,1+h*w,D) -> patch (1,D,h,w) -> (1,D,H,W) -> (1,H*W,D)。
例子：CLS=[99,99]，2×2 patch 网格换成 3×2 后 CLS 必须仍为 [99,99]。
常见错：把 embedding 的 D 轴当空间轴；输出 token 顺序与 patch 顺序不同。
接入：DINO 不同 crop 分辨率的 learned position embedding。
检查：python3 check_lessons.py --lesson 5；提示：HINTS.md 的基础 5。
"""
import torch
import torch.nn.functional as F


def interpolate_2d_pos_embed(position_embeddings, grid, *, num_prefix_tokens=1):
    # TODO 5.1：按 num_prefix_tokens 分离 prefix 与 patch positions。
    # TODO 5.2：从 patch 数推导旧方形网格，转为 (1,D,h,w)。
    # TODO 5.3：F.interpolate(..., size=grid, mode='bicubic', align_corners=False)。
    # TODO 5.4：转回 (1,H*W,D)，拼回不变的 prefix。
    raise NotImplementedError("基础 5.1–5.4：interpolate_2d_pos_embed")
