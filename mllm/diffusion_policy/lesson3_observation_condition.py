"""第 3 关：observation_condition。
前置：reshape/flatten 与视觉 backbone 的 batch 轴。局部检查提供其余关外围，完整 practice 需要全部适用关。
接入：practice_diffusion_policy.py 的 ops；网络模块初始化见 scaffolding.py 的类定义。
局部检查（仓库根）：PYTHONPATH=general_utils/edu_core python3 mllm/diffusion_policy/check_lessons.py --lesson 3
常见错误：不能打乱 B/To 顺序；视觉与 state 沿特征轴拼接；推理中不可反复编码。
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from edu_core import update_ema
IGNORE_INDEX = -100

def encode_observations(encoder, images, states):
    """images=(B,To,3,H,W)，states=(B,To,4)，输出 (B,To,16)。例如 B=2,To=2 时 encoder 收到 4 张图。
    TODO 3.1: 合并 B/To 轴做视觉编码。
    TODO 3.2: 恢复时序轴，沿特征轴拼 states；推理循环外调用。
    """
    raise NotImplementedError("lesson3.encode_observations: 完成本函数编号微任务")


