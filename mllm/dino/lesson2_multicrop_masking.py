"""关卡 2：构造两 global、四 local 与定量 mask

前置：裁剪与布尔张量。
形状合同：6 个 BCHW crops；mask [B,N] bool。
手算例子：N=4,ratio=0.5 每行恰好两处 True。
编号 TODO：1. 保持 globals 在前。2. 采样裁剪并 resize。3. 每行随机选固定数量 patch。
常见错误：不能把 local crop 当 teacher global；ratio 是本 toy 边长比例。
检查：python3 check_lessons.py --lesson 2 --implementation practice
提示：HINTS.md 第 2 关；参考检查可加 --implementation reference。
"""
from torch import Tensor


def make_multicrop_views(images: Tensor) -> list[Tensor]:
    """Return [global_0, global_1, local_0, ..., local_3]."""
    # TODO 1: 可复用 multicrop.random_resized_crop；两次 global_size=32,scale=(.5,1)。
    # TODO 2: 四次 local_size=16,scale=(.2,.5)；返回 globals + locals。
    raise NotImplementedError("lesson2.make_multicrop_views TODO 1-2")


def random_patch_mask(batch_size: int, num_patches: int, ratio: float, device) -> Tensor:
    """Return bool [B, N], ensuring every image has at least one True entry."""
    # TODO 3: count=max(1,round(N*ratio))；采样 [B,N] noise 每行 argsort。
    # TODO 4: bool zeros，用前 count 个索引 scatter_ True；保持输入 device。
    raise NotImplementedError("lesson2.random_patch_mask TODO 3-4")
