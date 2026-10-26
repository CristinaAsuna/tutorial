"""第 1 关：action_normalization。
前置：Tensor 归约、分位数、逐维广播。局部检查提供其余关外围，完整 practice 需要全部适用关。
接入：practice_openvla.py 的 ops；网络模块初始化见 scaffolding.py 的类定义。
局部检查（仓库根）：PYTHONPATH=general_utils/edu_core python3 mllm/openvla/check_lessons.py --lesson 1
常见错误：不能将验证集用于统计；mask=False 不缩放；clip 后不可严格反推离群原值。
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from edu_core import update_ema
IGNORE_INDEX = -100

def fit_action_stats(actions, mask):
    """只从训练动作拟合统计量。actions=(N,A)，mask=(A,)；N=101 等距数 0..100 的 q01=1、q99=99。
    TODO 1.1: 沿样本轴分别计算 .01 和 .99 分位数。
    TODO 1.2: 返回 q01/q99/mask 字典；mask 转 bool。
    """
    raise NotImplementedError("lesson1.fit_action_stats: 完成本函数编号微任务")


def normalize_actions(actions, stats):
    """物理动作转 [-1,1]。输入 (...,A)，输出同形；low=0 high=10 时 0,5,10 -> -1,0,1。
    TODO 1.1: 逐维缩放；分母至少 1e-6，常量维不产生 NaN。
    TODO 1.2: 裁剪缩放值，mask=False 的维度保留原值。
    """
    raise NotImplementedError("lesson1.normalize_actions: 完成本函数编号微任务")


def unnormalize_actions(actions, stats):
    """归一化动作恢复物理单位；-1,0,1 -> 0,5,10。
    TODO 1.1: 反解仿射变换。
    TODO 1.2: 仅恢复 mask=True 的维度；夹爪维度保持原值。
    """
    raise NotImplementedError("lesson1.unnormalize_actions: 完成本函数编号微任务")


