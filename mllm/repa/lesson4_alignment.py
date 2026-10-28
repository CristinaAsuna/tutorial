"""第4关：逐patch负余弦与联合目标。
输入projected/targets=(B,N,D)，归一化D轴后内积得到(B,N)，再mean。
手算[1,0]对[1,0]/[0,1]/[-1,0]：alignment分别-1/0/+1。
denoising MSE=4、alignment=-1、lambda=.5 -> total=3.5。
前置：normalize/MSE与stop-gradient；常见错：正cos、跨patch平均再对齐、teacher反传。
接入system.forward；alignment是有符号标量，不要求总loss为正。
检查：python3 -m mllm.repa.check_lessons --lesson 4。
"""
import torch
from torch.nn import functional as F


def alignment_loss(projected, targets):
    # TODO 4.1：检查相同(B,N,D)，targets.detach()，两者dim=-1归一化。
    # TODO 4.2：逐位置内积后取负，对B和N求均值。
    raise NotImplementedError('REPA lesson4.alignment_loss TODO 4.1-4.2')


def joint_objective(prediction, velocity, alignment, weight=.5):
    # TODO 4.3：MSE(prediction,velocity)，全部元素mean。
    # TODO 4.4：返回dict(loss=mse+weight*alignment,velocity_loss=mse,alignment_loss=alignment)。
    raise NotImplementedError('REPA lesson4.joint_objective TODO 4.3-4.4')
