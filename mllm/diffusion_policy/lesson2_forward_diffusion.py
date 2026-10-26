"""第 2 关：forward_diffusion。
前置：高斯重参数化、cumprod、batch gather。局部检查提供其余关外围，完整 practice 需要全部适用关。
接入：practice_diffusion_policy.py 的 ops；网络模块初始化见 scaffolding.py 的类定义。
局部检查（仓库根）：PYTHONPATH=general_utils/edu_core python3 mllm/diffusion_policy/check_lessons.py --lesson 2
常见错误：不能把 beta 当 alpha_bar；(B,) 系数须扩成 (B,1,1)；不能错用环境时间。
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from edu_core import update_ema
IGNORE_INDEX = -100

def cosine_schedule(steps):
    """K -> betas=(K,), alpha_bar=(K,)。f(t)=cos²((t+.008)/1.008*pi/2)；beta_i=1-f((i+1)/K)/f(i/K)。
    TODO 2.1: float64 构造 K+1 个累计余弦点并除首值。
    TODO 2.2: 相邻比值得 beta，最大 .999，转 float32。
    TODO 2.3: 对 1-beta 做 cumprod 得实际 alpha_bar。
    """
    raise NotImplementedError("lesson2.cosine_schedule: 完成本函数编号微任务")


def add_noise(clean, noise, timesteps, alpha_bar):
    """a_t=sqrt(alpha_bar_t)*a0+sqrt(1-alpha_bar_t)*eps。a0=2,eps=0,alpha_bar=.25 -> 1。
    TODO 2.1: 按每个 batch 的 timesteps gather 累计系数。
    TODO 2.2: reshape (B,1,1) 广播到 (B,Tp,A)。
    TODO 2.3: 组合 clean 与 noise。
    """
    raise NotImplementedError("lesson2.add_noise: 完成本函数编号微任务")


