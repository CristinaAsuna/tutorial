"""第 1 关：trajectory_windows。
前置：时间轴索引、单 episode 数据与逐维广播。局部检查提供其余关外围，完整 practice 需要全部适用关。
接入：practice_diffusion_policy.py 的 ops；网络模块初始化见 scaffolding.py 的类定义。
局部检查（仓库根）：PYTHONPATH=general_utils/edu_core python3 mllm/diffusion_policy/check_lessons.py --lesson 1
常见错误：起点不是当前 t 而是 t-To+1；不能跨 episode；重复填充不是 zero padding。
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from edu_core import update_ema
IGNORE_INDEX = -100

def trajectory_window(observations, actions, now, obs_horizon=2, pred_horizon=8):
    """单 episode，now=0,To=2,Tp=4 时观测索引 [0,0]、动作索引 [0,0,1,2]。
    TODO 1.1: 窗口起点 now-To+1。
    TODO 1.2: 观测取 To 步，动作取 Tp 步；分别 clamp 到同一 episode 有效索引。
    TODO 1.3: 重复首尾帧，返回 obs/action；不是 ACT 的未来窗口起点。
    """
    raise NotImplementedError("lesson1.trajectory_window: 完成本函数编号微任务")


def fit_bounds(data):
    """训练集 (N,A) -> 每维 low/high；[[0,4],[2,8]] -> low=[0,4],high=[2,8]。
    TODO 1.1: 沿样本轴取最小/最大值。
    TODO 1.2: 返回 low/high 字典，不能混入评估轨迹。
    """
    raise NotImplementedError("lesson1.fit_bounds: 完成本函数编号微任务")


def normalize(data, stats):
    """(...,A) 缩放到训练边界 [-1,1]；0,5,10 -> -1,0,1。
    TODO 1.1: 分母 clamp_min(1e-6)。
    TODO 1.2: 仿射缩放，不额外裁剪评估输入。
    """
    raise NotImplementedError("lesson1.normalize: 完成本函数编号微任务")


def unnormalize(data, stats):
    """归一化动作 -> 物理动作；low=0 high=10，0 -> 5。
    TODO 1.1: 反解归一化公式，保留广播轴。
    """
    raise NotImplementedError("lesson1.unnormalize: 完成本函数编号微任务")


