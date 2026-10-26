"""第5关 flow_matching：先读 README 对应数据流。
前置：基础 Tensor/attention；组合函数依赖前序关，单关检查提供外围参考实现。
已初始化模块与属性在 scaffolding.py，网络骨架无需学生重新设计。
局部命令（仓库根）：PYTHONPATH=general_utils/edu_core python3 mllm/pi0/check_lessons.py --lesson 5
常见错误：混淆环境时间与flow时间、反转allow mask、使prefix看见actions、缓存被不断追加。
"""
import torch
from edu_core.backbones import PrefixEncoding
from edu_core.backbones.layers import projected_attention


def sample_time(batch_size, device):
    """openpi时间约定：1为噪声，0为动作。time=.999*Beta(1.5,1)+.001，输出(B,)。
    TODO 5.1: 在指定device构造Beta分布。
    TODO 5.2: 采样batch_size个时间。
    TODO 5.3: 缩放到(.001,1)，不使用环境时间t。
    """
    raise NotImplementedError("lesson5.sample_time: 完成本函数编号任务")


def flow_pair(actions, noise, time):
    """x_t=t*eps+(1-t)*a，target=eps-a；a=2,eps=0,t=.25 -> x=1.5,target=-2。
    TODO 5.1: time扩成(B,1,1)。
    TODO 5.2: 线性混合动作与噪声。
    TODO 5.3: 返回插值和速度目标。
    """
    raise NotImplementedError("lesson5.flow_pair: 完成本函数编号任务")


def flow_loss(prediction, target):
    """prediction/target=(B,H,A)，MSE为所有动作元素的均值；[1,3]对[0,1] -> 2.5。
    TODO 5.1: 逐元素差平方。
    TODO 5.2: 全部轴求均值，不对state token或文字logits求此loss。
    """
    raise NotImplementedError("lesson5.flow_loss: 完成本函数编号任务")


