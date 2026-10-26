"""第1关：先画一个确定矩形，再接入已提供的随机 block-union 采样器。
前置：DINO 第2关，bool 索引。输入 grid=(H,W)，box=(top,left,h,w)。
手算：2x3 网格，box=(0,1,2,1) 得 [[F,T,F],[F,T,F]]。
形状：H,W -> HxW bool -> 外围采样器展平为 B,N；True 表示遮挡。
接入：model.views -> sample_masks -> 本关两个 helper。
常见错误：把像素坐标当 patch 坐标；未选中样本仍至少遮1个。
检查：python3 -m mllm.dinov2.check_lessons --lesson 1
"""

def rectangle_mask(grid, box, device=None):
    # TODO 1.1: 校验矩形边界；创建 HxW bool zeros。
    # TODO 1.2: 将 top:top+h, left:left+w 切片设 True，返回 mask。
    raise NotImplementedError('lesson1.rectangle_mask TODO 1.1-1.2')


def mask_count(num_patches, selected, ratio):
    # TODO 1.3: 未被选中返回0，否则 max(1,round(N*ratio))。
    # 手算 N=16，selected=True，ratio=.25 ->4；False ->0。
    raise NotImplementedError('lesson1.mask_count TODO 1.3')
