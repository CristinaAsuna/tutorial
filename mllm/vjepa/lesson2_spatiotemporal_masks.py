"""V-JEPA 关卡 2：采样互不重叠的时空 target blocks。"""
import torch


def make_masks(batch: int, grid: tuple[int, int, int], block: tuple[int, int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 target/context bool masks；context 必须严格等于 `~target`。"""
    # TODO: 在 T×H×W 网格上放置 block，并检查相交；最后 flatten。
    raise NotImplementedError("TODO: 实现无泄漏的时空 mask")
