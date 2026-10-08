"""基础 1：无损图像切片和还原。

目标：理解像素怎样变为 token，而不是学习卷积权重。
前置：reshape 保留元素次序；permute 交换轴；轴从 0 编号。
形状：B,C,H,W -> B,h,p,w,p,C -> B,h*w,p*p*C，h=H/p。
例子：单通道 4×4 的 arange(16)，p=2 时第一个 patch=[0,1,4,5]。
常见错：直接 reshape 成 B,N,D 会把不同 patch 的像素混到一起。
接入：完成后各论文默认复用 edu_core.vision.patchify/PatchEmbed。
检查：python3 check_lessons.py --lesson 1；提示：HINTS.md 的基础 1。
"""
import torch


def patchify(images: torch.Tensor, patch_size: int = 2) -> torch.Tensor:
    if images.ndim != 4 or patch_size <= 0:
        raise ValueError("需要 BCHW 和正 patch_size")
    b, c, height, width = images.shape
    p = patch_size
    if height % p or width % p:
        raise ValueError("H/W 必须整除 patch_size")
    h, w = height // p, width // p
    # TODO 1.1：拆出两个局部像素轴，得到 (B,C,h,p,w,p)。
    # TODO 1.2：将网格轴移到前面，局部轴变为 (p,p,C)。
    # TODO 1.3：合并网格与局部像素轴，返回 (B,h*w,p*p*C)。
    raise NotImplementedError("基础 1.1–1.3：patchify")


def unpatchify(patches: torch.Tensor, patch_size: int, channels: int, *, grid: tuple[int, int]):
    b, n, d = patches.shape
    h, w = grid
    p, c = patch_size, channels
    if h*w != n or d != p*p*c:
        raise ValueError("grid、channels 与 patches 不匹配")
    # TODO 1.4：拆成 (B,h,w,p,p,C)，将轴排列回 (B,C,h,p,w,p)。
    # TODO 1.5：合并相邻的网格轴/局部轴，返回 BCHW。
    raise NotImplementedError("基础 1.4–1.5：unpatchify")
