"""关卡 2：构造两 global、四 local 与定量 mask

前置：裁剪与布尔张量。
形状合同：6 个 BCHW crops；mask [B,N] bool。
手算例子：N=4,ratio=0.5 每行恰好两处 True。
编号 TODO：1a–1b. 按给定box逐样本裁剪resize。3–4. 每行随机选固定数量patch；global/local采样外围已给。
常见错误：不能把 local crop 当 teacher global；crop scale 是面积比例，mask ratio 是遮挡 patch 比例。
检查：python3 check_lessons.py --lesson 2 --implementation practice
提示：HINTS.md 第 2 关；参考检查可加 --implementation reference。
"""
from torch import Tensor


def crop_given_boxes(images: Tensor, boxes: Tensor, output_size: int) -> Tensor:
    """微任务：boxes [B,4]=(top,left,height,width)，同尺寸时可直接对照切片。

    手算：图像 [[0,1],[2,3]]，box=(0,1,2,1) 选择右列 [1,3]。
    """
    # TODO 1a: 按每行box切出该样本的CHW区域（校验边界）。
    # TODO 1b: 增加batch轴，bilinear resize(align_corners=False)，最后cat。
    raise NotImplementedError("lesson2.crop_given_boxes TODO 1a-1b")


def make_multicrop_views(images: Tensor) -> list[Tensor]:
    """已提供随机采样外围，只练确定性裁剪；globals始终在前。"""
    from multicrop import sample_crop_boxes
    settings = [(32,(.5,1.))]*2 + [(16,(.2,.5))]*4
    return [crop_given_boxes(images,sample_crop_boxes(images,scale),size) for size,scale in settings]


def random_patch_mask(batch_size: int, num_patches: int, ratio: float, device) -> Tensor:
    """Return bool [B, N], ensuring every image has at least one True entry."""
    # TODO 3: count=max(1,round(N*ratio))；采样 [B,N] noise 每行 argsort。
    # TODO 4: bool zeros，用前 count 个索引 scatter_ True；保持输入 device。
    raise NotImplementedError("lesson2.random_patch_mask TODO 3-4")
