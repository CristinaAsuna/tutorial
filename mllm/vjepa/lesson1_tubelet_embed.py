"""V-JEPA 关卡 1：把视频 `[B,C,T,H,W]` 切成 tubelet token。"""
import torch


def tubelet_patchify(videos: torch.Tensor, tubelet: int, patch: int) -> torch.Tensor:
    """返回 `[B, (T/t)(H/p)(W/p), C*t*p*p]`，要求所有维度恰好整除。"""
    # TODO: reshape 为 [B,C,T/t,t,H/p,p,W/p,p]，permute 后展平每个 tubelet。
    raise NotImplementedError("TODO: 实现 3D tubelet patchify")
