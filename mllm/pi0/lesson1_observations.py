"""第1关 observations：先读 README 对应数据流。
前置：基础 Tensor/attention；组合函数依赖前序关，单关检查提供外围参考实现。
已初始化模块与属性在 scaffolding.py，网络骨架无需学生重新设计。
局部命令（仓库根）：PYTHONPATH=general_utils/edu_core python3 mllm/pi0/check_lessons.py --lesson 1
常见错误：混淆环境时间与flow时间、反转allow mask、使prefix看见actions、缓存被不断追加。
"""
import torch


def action_chunk(actions, now, horizon=8):
    """给定单条 episode 的 (T,A) 动作，返回从当前 now 开始的 (H,A)。[0,1,2],now=1,H=4 -> [1,2,2,2]。
    TODO 1.1: 构造 now 到 now+H 的索引。
    TODO 1.2: 上界截断到 episode 末帧。
    TODO 1.3: 按索引取动作，不跨 episode。
    """
    raise NotImplementedError("lesson1.action_chunk: 完成本函数编号任务")


def normalize_values(values, low, high):
    """物理量 (...,A) -> [-1,1]；low=0,high=10 时 0,5,10 -> -1,0,1。统计来自训练集。
    TODO 1.1: 分母至少 1e-6。
    TODO 1.2: 逐维仿射缩放。
    TODO 1.3: 截断离群值，保留广播轴。
    """
    raise NotImplementedError("lesson1.normalize_values: 完成本函数编号任务")


def unnormalize_values(values, low, high):
    """归一化动作恢复物理单位；-1,0,1 -> 0,5,10。输入输出形状相同。
    TODO 1.1: 反解仿射式。
    TODO 1.2: 不要对最终物理输出再次乘机器人控制频率。
    """
    raise NotImplementedError("lesson1.unnormalize_values: 完成本函数编号任务")
