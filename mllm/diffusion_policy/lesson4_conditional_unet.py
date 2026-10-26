"""第 4 关：conditional_unet。
前置：Conv1d 通道顺序、残差、sin/cos；条件向量。局部检查提供其余关外围，完整 practice 需要全部适用关。
接入：practice_diffusion_policy.py 的 ops；网络模块初始化见 scaffolding.py 的类定义。
局部检查（仓库根）：PYTHONPATH=general_utils/edu_core python3 mllm/diffusion_policy/check_lessons.py --lesson 4
常见错误：scale*x+bias 不是 (1+scale)*x；skip 拼通道而非时间；上采样后需恢复原 Tp。
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from edu_core import update_ema
IGNORE_INDEX = -100

def timestep_embedding(timesteps, dim):
    """时间步 (B,) -> (B,D)，D 为偶数且 >=4；t=0 时 sin 半边全0、cos 半边全1。
    TODO 4.1: 构造半维指数频率，范围 1..1/10000。
    TODO 4.2: 时间步外积频率，拼 sin/cos。
    """
    raise NotImplementedError("lesson4.timestep_embedding: 完成本函数编号微任务")


def film_modulate(features, scale_bias):
    """x=(B,C,T)，scale_bias=(B,2C)。scale=2,bias=3,x=1 -> 5，使用 scale*x+bias。
    TODO 4.1: 沿通道拆 scale/bias。
    TODO 4.2: 末尾加时间广播轴；不是 (1+scale)*x。
    """
    raise NotImplementedError("lesson4.film_modulate: 完成本函数编号微任务")


def residual_forward(block, x, condition):
    """初始化在 ConditionalResidual 中。conv1 -> FiLM -> conv2 + skip；输入 (B,Cin,T)。
    TODO 4.1: 运行 block.conv1。
    TODO 4.2: 调用 block.ops.film_modulate 和 block.condition。
    TODO 4.3: 运行 conv2，加投影后的残差。
    """
    raise NotImplementedError("lesson4.residual_forward: 完成本函数编号微任务")


def unet_down(model, x, cond):
    """输入 (B,2,8) -> skip (B,16,8)、down (B,32,4)。
    TODO 4.1: 条件残差块提取 skip。
    TODO 4.2: downsample 缩短时间轴；返回 (down,skip)。
    """
    raise NotImplementedError("lesson4.unet_down: 完成本函数编号微任务")


def unet_up(model, x, skip, cond):
    """输入 (B,32,4)、skip (B,16,8) -> (B,16,8)。
    TODO 4.1: upsample 恢复时间轴。
    TODO 4.2: 沿通道拼 skip，再进条件残差块。
    """
    raise NotImplementedError("lesson4.unet_up: 完成本函数编号微任务")


def unet_forward(model, sample, timesteps, obs):
    """(B,Tp,A) -> 噪声预测同形；cond=(B,48)=16维时间+2*16维观测。
    TODO 4.1: 通过 model.ops.timestep_embedding 与 time_mlp 构造时间条件，拼 obs.flatten(1)。
    TODO 4.2: 转成 Conv1d 的 B,A,T；调用 ops.unet_down。
    TODO 4.3: middle 后调用 ops.unet_up。
    TODO 4.4: output 投影并转回 B,T,A。
    """
    raise NotImplementedError("lesson4.unet_forward: 完成本函数编号微任务")


