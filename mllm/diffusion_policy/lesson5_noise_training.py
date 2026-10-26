"""第 5 关：noise_training。
前置：噪声预测目标、梯度、共享 EMA helper。局部检查提供其余关外围，完整 practice 需要全部适用关。
接入：practice_diffusion_policy.py 的 ops；网络模块初始化见 scaffolding.py 的类定义。
局部检查（仓库根）：PYTHONPATH=general_utils/edu_core python3 mllm/diffusion_policy/check_lessons.py --lesson 5
常见错误：target 是实际采样的 epsilon；先 optimizer 后 EMA；EMA 无梯度，buffers 不由 helper 更新。
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from edu_core import update_ema
IGNORE_INDEX = -100

def noise_loss(prediction, noise):
    """预测 eps 与真实 eps 的 MSE；[1,3] 对 [0,1] -> (1+4)/2=2.5。
    TODO 5.1: 对每个元素平方误差，全部轴求均值；重复填充窗口同样参与 loss。
    """
    raise NotImplementedError("lesson5.noise_loss: 完成本函数编号微任务")


def training_step(model, ema, batch, optimizer, noise, timesteps, decay=0.95):
    """loss -> backward -> optimizer -> EMA。ema 参数 frozen；默认 decay=.95。
    TODO 5.1: train 与 zero_grad；model.compute_loss 接收固定 noise/timesteps。
    TODO 5.2: backward 后 step。
    TODO 5.3: 调用共享 update_ema(ema,model,decay)；返回 detach loss。
    """
    raise NotImplementedError("lesson5.training_step: 完成本函数编号微任务")


