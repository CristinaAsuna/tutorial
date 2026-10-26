"""第 2 关：action_tokenizer。
前置：排序搜索、long 索引、归一化动作范围。局部检查提供其余关外围，完整 practice 需要全部适用关。
接入：practice_openvla.py 的 ops；网络模块初始化见 scaffolding.py 的类定义。
局部检查（仓库根）：PYTHONPATH=general_utils/edu_core python3 mllm/openvla/check_lessons.py --lesson 2
常见错误：bucketize 的 right 语义与 numpy.digitize 相反；边界须用 float64；255 个中心并非256。
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from edu_core import update_ema
IGNORE_INDEX = -100

def encode_actions(actions, vocab_size=320, n_bins=256):
    """动作 (...,A) -> long IDs (...,A)。小例 n_bins=4,V=20：[-1,0,1] -> [19,18,16]。
    TODO 2.1: float64 arange*(2/(n_bins-1))-1 构造官方等距边界并裁剪动作；保留比较精度。
    TODO 2.2: 用等价于 numpy.digitize(right=False) 的 bucketize，注意 torch 的 right 参数。
    TODO 2.3: 返回 V-index，范围为 [V-n_bins,V-1]。
    """
    raise NotImplementedError("lesson2.encode_actions: 完成本函数编号微任务")


def decode_actions(ids, vocab_size=320, n_bins=256):
    """token IDs -> bin centers。四边界 [-1,-1/3,1/3,1] 的中心为 [-2/3,0,2/3]。
    TODO 2.1: 先计算相邻边界的中点。
    TODO 2.2: 将 V-id-1 裁剪至 [0,n_bins-2] 再索引；最后两个边界索引可共享中心。
    """
    raise NotImplementedError("lesson2.decode_actions: 完成本函数编号微任务")


