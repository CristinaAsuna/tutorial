"""第 4 关：multimodal_sequence。
前置：基础 mask、embedding、单行拼接。局部检查提供其余关外围，完整 practice 需要全部适用关。
接入：practice_openvla.py 的 ops；网络模块初始化见 scaffolding.py 的类定义。
局部检查（仓库根）：PYTHONPATH=general_utils/edu_core python3 mllm/openvla/check_lessons.py --lesson 4
常见错误：视觉插在 BOS 后；视觉/prompt/padding 标签均 ignore；不能原地修改 ids。
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from edu_core import update_ema
IGNORE_INDEX = -100

def pack_single(text_embeds, visual, labels):
    """单行插入视觉 token：文本 [BOS,7,8]、视觉 [v0,v1] -> [BOS,v0,v1,7,8]；视觉 labels=-100。
    TODO 4.1: 拼接文本首 token、视觉 embedding、剩余文本 embedding。
    TODO 4.2: 相同位置插入 ignore labels，返回 (embeds, labels)；不改变 labels 的监督值。
    """
    raise NotImplementedError("lesson4.pack_single: 完成本函数编号微任务")


def action_labels(ids, action_dim, predict_stop=True):
    """ids=BOS+指令+A 个动作+EOS，只监督动作和可选 EOS。例如 [1,8,64,65,2],A=2 -> [-100,-100,64,65,2]。
    TODO 4.1: clone ids，不原地更改输入。
    TODO 4.2: 屏蔽最后 A+1 个位置之前的所有标签。
    TODO 4.3: predict_stop=False 时屏蔽最后 EOS。
    """
    raise NotImplementedError("lesson4.action_labels: 完成本函数编号微任务")


