"""基础 2：把视频拆为时空 tubelet。

前置：基础 1；视频轴是 B,C,T,H,W，T 是帧数。
形状：(B,C,nt,t,nh,p,nw,p) -> (B,nt,nh,nw,t,p,p,C) -> (B,N,t*p*p*C)。
例子：两帧分别为 [[0,1],[2,3]] 与 [[4,5],[6,7]]，t=2,p=2，唯一 token 为 0..7。
常见错：帧轴与通道轴混淆；token 网格应宽度变化最快，时间最慢。
接入：V-JEPA 复用 edu_core.vision.TubeletEmbed，学习投影另由 Conv3d 完成。
检查：python3 check_lessons.py --lesson 2；提示：HINTS.md 的基础 2。
"""
import torch


def tubelet_patchify(videos: torch.Tensor, tubelet: int = 2, patch: int = 2):
    if videos.ndim != 5 or tubelet <= 0 or patch <= 0:
        raise ValueError("需要 BCTHW 和正 tubelet/patch")
    b, c, frames, height, width = videos.shape
    t, p = tubelet, patch
    if frames % t or height % p or width % p:
        raise ValueError("T/H/W 必须整除 tubelet/patch")
    nt, nh, nw = frames // t, height // p, width // p
    # TODO 2.1：拆成 (B,C,nt,t,nh,p,nw,p)。
    # TODO 2.2：按文档把网格轴、局部轴分组。
    # TODO 2.3：返回 (B,nt*nh*nw,t*p*p*C)。
    raise NotImplementedError("基础 2.1–2.3：tubelet_patchify")
