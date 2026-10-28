"""第4关：reconstruction。
前置：foundations patchify/attention/mask/EMA；各函数目的、形状和手算见下方。
接入：practice_ops把本关函数注入RAECourse；单关检查提供其他关参考外围。
检查（仓库根）：PYTHONPATH=general_utils/edu_core python3 -m mllm.rae.check_lessons --lesson 4
常见错误：对错轴归约、错时间方向、切断可训练梯度、把eval误当冻结。
"""
import torch
from torch.nn import functional as F
from edu_core.vision import unpatchify
from edu_core.training import update_ema

def reconstruction_loss(prediction, images):
    """目的：像素L1：该toy不冒充论文完整LPIPS/GAN目标。
    形状：两个(B,3,16,16) → 零维标量。
    手算：prediction=[1,3],target=[0,1] → 1.5。
    """
    # TODO 4.1.1: prediction-images得到逐像素误差。
    # TODO 4.1.2: 对误差绝对值在全部轴求mean，返回可反传的零维loss。
    raise NotImplementedError("RAE lesson4.reconstruction_loss TODO 4.1")


def stage1_step(course, images, optimizer):
    """目的：第一阶段只更新decoder，encoder保持frozen/eval。
    形状：images=(B,3,16,16) → reconstruction → scalar loss。
    手算：encoder=2不变；decoder参数按p←p-lr*grad更新。
    """
    # TODO 4.2.1: course.set_stage(1)，optimizer.zero_grad(set_to_none=True)。
    # TODO 4.2.2: prediction=course.rae(images)，调用course.ops.reconstruction_loss(prediction,images)。
    # TODO 4.2.3: loss.backward()后optimizer.step()，返回loss.detach()。
    raise NotImplementedError("RAE lesson4.stage1_step TODO 4.2")


