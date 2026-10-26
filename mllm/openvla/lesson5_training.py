"""第 5 关：training。
前置：交叉熵、autograd、optimizer；前序标签合同。局部检查提供其余关外围，完整 practice 需要全部适用关。
接入：practice_openvla.py 的 ops；网络模块初始化见 scaffolding.py 的类定义。
局部检查（仓库根）：PYTHONPATH=general_utils/edu_core python3 mllm/openvla/check_lessons.py --lesson 5
常见错误：shift 两次、对 prompt 算损失、漏掉 optimizer.step，都会破坏训练。
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from edu_core import update_ema
IGNORE_INDEX = -100

def shifted_loss(logits, labels):
    """logits=(B,L,V)、labels=(B,L)；位置 i 的预测对齐位置 i+1 的标签，不再做第二次 shift。
    TODO 5.1: 取 logits 去末位、labels 去首位。
    TODO 5.2: 展平 B 与 L，交叉熵 ignore_index=-100。
    """
    raise NotImplementedError("lesson5.shifted_loss: 完成本函数编号微任务")


def training_step(model, images, prompts, actions, stats, optimizer):
    """一次完整训练更新。model(images,prompts,actions,stats) 返回标量 loss。
    TODO 5.1: 进入 train，清零梯度。
    TODO 5.2: 调用 model 并反向传播；优化器 step。
    TODO 5.3: 返回 detach loss，不能重新计算参考 loss。
    """
    raise NotImplementedError("lesson5.training_step: 完成本函数编号微任务")


