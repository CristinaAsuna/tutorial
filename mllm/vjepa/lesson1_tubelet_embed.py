"""关卡 1：视频按 tubelet 展平

前置：基础 patchify/reshape/permute。
形状合同：[B,C,T,H,W] → [B,N,C*t*p*p]。
手算例子：[1,1,2,2,2] 的 arange(8),t=2,p=1 → [[0,4],[1,5],[2,6],[3,7]]。
编号 TODO：1. 校验整除。2. 拆开网格轴和块内轴。3. 把网格轴移前。4. flatten。
常见错误：token 顺序 time,height,width；块内顺序 time,height,width,channel（与共享基础关一致）。
检查：python3 check_lessons.py --lesson 1 --implementation practice
提示：HINTS.md 第 1 关；参考检查可加 --implementation reference。
"""
import torch


def tubelet_patchify(videos: torch.Tensor, tubelet: int, patch: int) -> torch.Tensor:
    """返回 `[B, (T/t)(H/p)(W/p), C*t*p*p]`，要求所有维度恰好整除。"""
    # TODO: reshape 为 [B,C,T/t,t,H/p,p,W/p,p]，permute 后展平每个 tubelet。
    raise NotImplementedError("TODO: 实现 3D tubelet patchify")
