"""第2关：decoder。
前置：foundations patchify/attention/mask/EMA；各函数目的、形状和手算见下方。
接入：practice_ops把本关函数注入RAECourse；单关检查提供其他关参考外围。
检查（仓库根）：PYTHONPATH=general_utils/edu_core python3 -m mllm.rae.check_lessons --lesson 2
常见错误：对错轴归约、错时间方向、切断可训练梯度、把eval误当冻结。
"""
import torch
from torch.nn import functional as F
from edu_core.vision import unpatchify
from edu_core.training import update_ema

def decode_tokens(model, latent):
    """目的：ViT decoder把表示恢复到像素；encoder和decoder的patch语义不同。
    形状：latent=(B,16,64) → decoder → pixels=(B,16,48) → (B,3,16,16)。
    手算：4×4网格每个token对应4×4×3=48个像素值；第0 token回到左上角。
    """
    # TODO 2.1.1: 将latent送入model.decoder_blocks，得到相同token数的hidden。
    # TODO 2.1.2: 用model.decoder_out把每token投影到4*4*3=48。
    # TODO 2.1.3: 调用unpatchify(...,4,3,grid=(4,4))还原RGB，不改变局部p,p,C顺序。
    raise NotImplementedError("RAE lesson2.decode_tokens TODO 2.1")


