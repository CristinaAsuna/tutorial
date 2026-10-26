"""第4关：同一图像区域，高分辨率教师提供更细网格，再对齐学生。
前置：DINO裁剪helper，网格宽度最快。
形状：同box -> low B,3,32,32 / high B,3,64,64；teacher B,64,D -> B,16,D。
手算：2x2常数feature网格降至1x1，值不变；token轴绝不能直接interpolate。
接入：model.views用aligned_crops；refinement前downsample_features；归一化在其后Gram中。
常见错误：low/high独立随机crop；先归一化再resize；H/W展平顺序反了。
检查：python3 -m mllm.dinov3.check_lessons --lesson 4
"""

def aligned_crops(images, boxes, low_size=32, high_size=64):
    # TODO 4.1: 调用 mllm.dino.multicrop.crop_and_resize，复用同一boxes两次。
    raise NotImplementedError('lesson4.aligned_crops TODO 4.1')


def downsample_features(features, source_grid, target_grid):
    # TODO 4.2: B,N,D -> B,D,Hs,Ws；N应等于Hs*Ws。
    # TODO 4.3: bicubic interpolate(size=target_grid,align_corners=False,antialias=False)。
    # TODO 4.4: flatten空间并transpose，返回B,Ntarget,D，不在此normalize。
    raise NotImplementedError('lesson4.downsample_features TODO 4.2-4.4')
