"""第 6 关：reverse_sampling。
前置：第2关的累计系数、DDPM posterior、固定噪声。局部检查提供其余关外围，完整 practice 需要全部适用关。
接入：practice_diffusion_policy.py 的 ops；网络模块初始化见 scaffolding.py 的类定义。
局部检查（仓库根）：PYTHONPATH=general_utils/edu_core python3 mllm/diffusion_policy/check_lessons.py --lesson 6
常见错误：t=0 必须不加噪；DDIM previous=-1 对应干净输出；图像编码在循环外。
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from edu_core import update_ema
IGNORE_INDEX = -100

def ddpm_step(sample, prediction, t, betas, alpha_bar, noise):
    """t 是单个整数，noise 与 sample 同形。令 a=alpha_bar[t],p=alpha_bar[t-1]，t=0 时 p=1。
    TODO 6.1: 从 eps 反推 x0=(sample-sqrt(1-a)*eps)/sqrt(a)，裁剪 [-1,1]。
    TODO 6.2: 均值 sqrt(p)*beta/(1-a)*x0 + sqrt(1-beta)*(1-p)/(1-a)*sample。
    TODO 6.3: 方差 beta*(1-p)/(1-a)，仅 t>0 加噪。
    """
    raise NotImplementedError("lesson6.ddpm_step: 完成本函数编号微任务")


def ddim_step(sample, prediction, t, previous_t, alpha_bar):
    """eta=0 的确定性跳步；previous_t=-1 时返回最终预测 x0。
    TODO 6.1: 还原并裁剪 x0。
    TODO 6.2: previous_t<0 时累计系数取1。
    TODO 6.3: sqrt(a_prev)*x0 + sqrt(1-a_prev)*prediction。
    """
    raise NotImplementedError("lesson6.ddim_step: 完成本函数编号微任务")


def sample_actions(model, images, states, initial_noise, step_noises, sampler='ddpm', ddim_steps=4):
    """返回归一化 (B,Tp,A)；images/states 仅在循环前编码一次。
    TODO 6.1: 用 model.ops.encode_observations 缓存条件；clone 初始噪声。
    TODO 6.2: DDPM 从 K-1 到0；每步构造 (B,) 时间，denoiser 预测后调用 ops.ddpm_step。
    TODO 6.3: DDIM 校验 ddim_steps，使用 linspace(K-1,0) 的递减网格和 ops.ddim_step。
    TODO 6.4: 未知 sampler 报 ValueError；不能重新编码图像。
    """
    raise NotImplementedError("lesson6.sample_actions: 完成本函数编号微任务")


