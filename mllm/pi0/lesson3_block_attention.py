"""第3关 block_attention：先读 README 对应数据流。
前置：基础 Tensor/attention；组合函数依赖前序关，单关检查提供外围参考实现。
已初始化模块与属性在 scaffolding.py，网络骨架无需学生重新设计。
局部命令（仓库根）：PYTHONPATH=general_utils/edu_core python3 mllm/pi0/check_lessons.py --lesson 3
常见错误：混淆环境时间与flow时间、反转allow mask、使prefix看见actions、缓存被不断追加。
"""
import torch
from edu_core.backbones import PrefixEncoding
from edu_core.backbones.layers import projected_attention


def block_mask(valid, prefix_length):
    """valid=(B,P+1+H)，分组 prefix=0,state=1,actions=2。P=1,H=2 时允许矩阵为 [[1,0,0,0],[1,1,0,0],[1,1,1,1],[1,1,1,1]]。
    TODO 3.1: 构造三组编号。
    TODO 3.2: key组编号 <= query组编号时允许。
    TODO 3.3: 同时屏蔽无效query与key，True=允许。
    """
    raise NotImplementedError("lesson3.block_mask: 完成本函数编号任务")


def position_ids(valid):
    """valid=[1,1,0,1] -> positions=[0,1,1,2]，padding不增加实际位置。
    TODO 3.1: valid转long做cumsum。
    TODO 3.2: 减1，左侧padding处负位置截断到0。
    """
    raise NotImplementedError("lesson3.position_ids: 完成本函数编号任务")


