"""第 8 关：transformer_extension。
前置：基础 self/cross attention、位置编码、allow mask。局部检查提供其余关外围，完整 practice 需要全部适用关。
接入：practice_diffusion_policy.py 的 ops；网络模块初始化见 scaffolding.py 的类定义。
局部检查（仓库根）：PYTHONPATH=general_utils/edu_core python3 mllm/diffusion_policy/check_lessons.py --lesson 8
常见错误：扩散时间在 memory 第0列；True=允许；不能让 action0 看 o1 或未来动作；并行预测 epsilon 不是逐维生成。
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from edu_core import update_ema
IGNORE_INDEX = -100

def transformer_masks(horizon, obs_horizon, device=None):
    """True=允许注意；动作因果三角。memory 列为 [diffusion_time,o0,o1]，action0 可见 [True,True,False]。
    TODO 8.1: 生成 (Tp,Tp) 下三角 allow mask。
    TODO 8.2: cross mask[i,j]=(i>=j-1)，扩散时间列总可见。
    """
    raise NotImplementedError("lesson8.transformer_masks: 完成本函数编号微任务")


def transformer_tokens(model, sample, timesteps, obs):
    """sample=(B,8,2)->(B,8,32)，obs=(B,2,16)->memory=(B,3,32)。
    TODO 8.1: 投影动作并加 action_pos。
    TODO 8.2: 时间 embedding 经 time_mlp 后加 token 轴。
    TODO 8.3: 拼时间与 obs_in(obs)，加 memory_pos。
    """
    raise NotImplementedError("lesson8.transformer_tokens: 完成本函数编号微任务")


def transformer_forward(model, sample, timesteps, obs):
    """两层因果 self-attention + 时间对齐 cross-attention，输出 (B,8,2)。
    TODO 8.1: 调用 ops.transformer_tokens 和 ops.transformer_masks。
    TODO 8.2: 每层 pre-LN self attention 残差，再带 cross_mask 的 cross attention 残差。
    TODO 8.3: MLP 残差后最终 norm/output；True 表示允许，不要反转。
    """
    raise NotImplementedError("lesson8.transformer_forward: 完成本函数编号微任务")


