"""第2关 prefix_suffix：先读 README 对应数据流。
前置：基础 Tensor/attention；组合函数依赖前序关，单关检查提供外围参考实现。
已初始化模块与属性在 scaffolding.py，网络骨架无需学生重新设计。
局部命令（仓库根）：PYTHONPATH=general_utils/edu_core python3 mllm/pi0/check_lessons.py --lesson 2
常见错误：混淆环境时间与flow时间、反转allow mask、使prefix看见actions、缓存被不断追加。
"""
import torch
from edu_core.backbones import PrefixEncoding
from edu_core.backbones.layers import projected_attention


def embed_prefix(model, images, camera_valid, text_ids, text_valid):
    """images=(B,M,3,H,W)，camera_valid=(B,M)，text=(B,L)，输出 (B,M*N+L,Dv) 和有效位。两相机各4个tokens + 3文字 -> 11。
    TODO 2.1: 合并 B/M 调用 backbone.encode_vision(projected=True)。
    TODO 2.2: 恢复相机次序，并展开相机有效位结合视觉mask。
    TODO 2.3: 拼视觉与 backbone.embed_text，拼对应有效mask。
    """
    raise NotImplementedError("lesson2.embed_prefix: 完成本函数编号任务")


def time_embedding(time, dim):
    """time=(B,)，dim 为偶数。period=.004*(4/.004)^fraction；time=0 时 sin 半边为0，cos 半边为1。
    TODO 2.1: 构造 dim/2 个 0到1 的 fraction。
    TODO 2.2: 计算跨 .004 到4的几何周期。
    TODO 2.3: 按 2*pi*time/period 拼 sin 和 cos。
    """
    raise NotImplementedError("lesson2.time_embedding: 完成本函数编号任务")


def embed_suffix(model, state, noisy_actions, time):
    """state=(B,7)，noisy=(B,8,7)，输出 (B,9,De)：一个state token、八个动作token。
    TODO 2.1: 使用 model.state_proj 生成单个state token。
    TODO 2.2: 动作经过 action_in，时间经 model.ops.time_embedding 并沿H扩展。
    TODO 2.3: 拼动作/时间特征，送 action_time，前置state token。
    """
    raise NotImplementedError("lesson2.embed_suffix: 完成本函数编号任务")


