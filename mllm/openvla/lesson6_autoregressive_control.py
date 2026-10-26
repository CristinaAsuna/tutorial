"""第 6 关：autoregressive_control。
前置：前序多模态 packing 与动作 tokenizer；因果LM。局部检查提供其余关外围，完整 practice 需要全部适用关。
接入：practice_openvla.py 的 ops；网络模块初始化见 scaffolding.py 的类定义。
局部检查（仓库根）：PYTHONPATH=general_utils/edu_core python3 mllm/openvla/check_lessons.py --lesson 6
常见错误：不能读取 batch 的公共末位；每行须取最后有效位置；不能用 teacher-forced logits 冒充生成。
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from edu_core import update_ema
IGNORE_INDEX = -100

def generate_action_ids(model, images, prompts):
    """从不同长度 prompts 逐个生成 A 维动作。B=2，prompt 长度 2/3；第一次分别读各自最后有效 token 的 logits。
    TODO 6.1: 初始化行副本；视觉只编码一次。
    TODO 6.2: 循环 A 次，调用 model.pack 与 model.language。
    TODO 6.3: 按有效长度取得最后 logits，全词表 argmax；追加行尾。
    TODO 6.4: 按动作维堆叠生成结果；不额外强制 action-only 词表。
    """
    raise NotImplementedError("lesson6.generate_action_ids: 完成本函数编号微任务")


def predict_action(model, images, prompts, stats):
    """ID (B,A) -> 归一化动作 -> 物理动作。非法文本 ID 按官方 decoder 的裁剪规则处理。
    TODO 6.1: 通过 model.ops 调用生成函数。
    TODO 6.2: 通过 model.ops 解码并按 stats 反归一化；不得直接调用参考函数。
    """
    raise NotImplementedError("lesson6.predict_action: 完成本函数编号微任务")


