"""第6关：ddt。
前置：foundations patchify/attention/mask/EMA；各函数目的、形状和手算见下方。
接入：practice_ops把本关函数注入RAECourse；单关检查提供其他关参考外围。
检查（仓库根）：PYTHONPATH=general_utils/edu_core python3 -m mllm.rae.check_lessons --lesson 6
常见错误：对错轴归约、错时间方向、切断可训练梯度、把eval误当冻结。
"""
import torch
from torch.nn import functional as F
from edu_core.vision import unpatchify
from edu_core.training import update_ema

def ddt_forward(model, noisy, time):
    """目的：窄base供逐token条件，宽head独立读取原始噪声latent，绕过窄投影。
    形状：noisy=(B,16,64)，base=(B,16,32)，condition/head=(B,16,96)，velocity=(B,16,64)。
    手算：identity projector、t=2、两token的s=[1,3]时，condition=SiLU([3,5])≈[2.858,4.967]；先pool则变成重复SiLU(4)，丢失位置差异。
    """
    # TODO 6.1.1: t=model.time_embed(time[:,None])；s=model.base_embed(noisy)+model.pos。
    # TODO 6.1.2: 逐个运行model.base中的block(s,F.silu(t))，保留B,N,32。
    # TODO 6.1.3: condition=model.condition_projector(F.silu(t[:,None]+s))，得到B,N,96，不pool token轴。
    # TODO 6.1.4: 独立宽路径x=model.noisy_embed(noisy)，必须从原始64维noisy进入，不能使用窄base的s。
    # TODO 6.1.5: 逐个运行model.head中的block(x,condition)，AdaLN外围已提供。
    # TODO 6.1.6: shift,scale=model.final_modulation(condition).chunk(2,-1)；out(final_norm(x)*(1+scale)+shift)。
    raise NotImplementedError("RAE lesson6.ddt_forward TODO 6.1")


