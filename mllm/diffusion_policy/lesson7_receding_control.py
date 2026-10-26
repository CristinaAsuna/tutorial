"""第 7 关：receding_control。
前置：第1关 horizon 对齐、第6关整段采样。局部检查提供其余关外围，完整 practice 需要全部适用关。
接入：practice_diffusion_policy.py 的 ops；网络模块初始化见 scaffolding.py 的类定义。
局部检查（仓库根）：PYTHONPATH=general_utils/edu_core python3 mllm/diffusion_policy/check_lessons.py --lesson 7
常见错误：执行起点 To-1；执行 Ta 步再更新观测；不混用 ACT 的 temporal ensemble。
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from edu_core import update_ema
IGNORE_INDEX = -100

def execution_chunk(prediction, obs_horizon=2, action_horizon=3):
    """预测 [a0,a1,...,a7]，To=2,Ta=3 -> [a1,a2,a3]。
    TODO 7.1: 从 To-1 起沿时间轴取 Ta 步，保留 B/A 轴。
    """
    raise NotImplementedError("lesson7.execution_chunk: 完成本函数编号微任务")


def receding_rollout(model, observe, execute, rounds, initial_noises, step_noises):
    """observe()->(images,states)，execute(chunk) 更新环境及历史；每轮执行 Ta 步才重规划。
    TODO 7.1: 每轮获取最新观测。
    TODO 7.2: model.predict_action 接收对应轮次的固定噪声。
    TODO 7.3: execute(result[action]) 并收集 chunk，不做 ACT temporal ensemble。
    """
    raise NotImplementedError("lesson7.receding_rollout: 完成本函数编号微任务")


