"""第 3 关：fused_vision。
前置：基础 patch tokens、最后一轴线性映射。局部检查提供其余关外围，完整 practice 需要全部适用关。
接入：practice_openvla.py 的 ops；网络模块初始化见 scaffolding.py 的类定义。
局部检查（仓库根）：PYTHONPATH=general_utils/edu_core python3 mllm/openvla/check_lessons.py --lesson 3
常见错误：沿 N 拼接会改变 patch 对齐；不能冻结或 detach 任一视觉分支。
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from edu_core import update_ema
IGNORE_INDEX = -100

def fuse_features(first, second, projector):
    """两路 (B,N,D1)/(B,N,D2) 沿特征轴拼接。[[1,2]] + [[3]] -> [[1,2,3]]。
    TODO 3.1: 检查 N 是相同网格，不是两个不同序列。
    TODO 3.2: 沿最后一轴拼接，再调用已初始化的 projector。
    """
    raise NotImplementedError("lesson3.fuse_features: 完成本函数编号微任务")


