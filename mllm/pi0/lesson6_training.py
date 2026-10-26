"""第6关 training：先读 README 对应数据流。
前置：基础 Tensor/attention；组合函数依赖前序关，单关检查提供外围参考实现。
已初始化模块与属性在 scaffolding.py，网络骨架无需学生重新设计。
局部命令（仓库根）：PYTHONPATH=general_utils/edu_core python3 mllm/pi0/check_lessons.py --lesson 6
常见错误：混淆环境时间与flow时间、反转allow mask、使prefix看见actions、缓存被不断追加。
"""
import torch
from edu_core.backbones import PrefixEncoding
from edu_core.backbones.layers import projected_attention


def training_step(model, batch, optimizer, noise=None, time=None):
    """一次更新：zero_grad -> model(batch) -> backward -> step。tiny默认端到端梯度，冻结预训练模式只更新expert。
    TODO 6.1: 调用model.train并清梯度。
    TODO 6.2: 调用model，传入显式noise/time用于可重复检查。
    TODO 6.3: 反向传播并optimizer.step。
    TODO 6.4: 返回detach loss。
    """
    raise NotImplementedError("lesson6.training_step: 完成本函数编号任务")


