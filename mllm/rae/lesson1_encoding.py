"""第1关：encoding。
前置：foundations patchify/attention/mask/EMA；各函数目的、形状和手算见下方。
接入：practice_ops把本关函数注入RAECourse；单关检查提供其他关参考外围。
检查（仓库根）：PYTHONPATH=general_utils/edu_core python3 -m mllm.rae.check_lessons --lesson 1
常见错误：对错轴归约、错时间方向、切断可训练梯度、把eval误当冻结。
"""
import torch
from torch.nn import functional as F
from edu_core.vision import unpatchify
from edu_core.training import update_ema

def normalize_tokens(tokens, special_tokens=2):
    """目的：去掉特殊token，保留空间对应的patch；逐token无affine LN。
    形状：tokens=(B,S+N,D) → (B,N,D)，沿D求均值/方差，不跨B/N。
    手算：[CLS,REG,[1,3]] → [-1,1]（忽略eps）。
    """
    # TODO 1.1.1: 先验证B,N,D和special_tokens范围；沿token轴取tokens[:,special_tokens:]。
    # TODO 1.1.2: 对每个patch最后D轴做F.layer_norm(...,(D,),weight=None,bias=None)，不跨batch/token。
    # TODO 1.1.3: 返回归一化patches；保留B,N次序。
    raise NotImplementedError("RAE lesson1.normalize_tokens TODO 1.1")


def tokens_to_grid(tokens, grid=(4, 4)):
    """形状(B,N,D)→(B,D,H,W)；编号tokens=[0,1,2,3]按2×2变成[[0,1],[2,3]]，宽度最快。
    TODO 1.2.1 校验N=H*W；1.2.2 transpose token/channel轴后reshape。
    接入：stage2用rae.encode_grid，再grid_to_tokens送生成器。
    """
    raise NotImplementedError("RAE lesson1.tokens_to_grid TODO 1.2")


def grid_to_tokens(grid):
    """形状(B,D,H,W)→(B,H*W,D)；手算是上一函数逆过程。
    TODO 1.3.1 flatten空间轴；1.3.2 transpose还原token轴在中间。
    """
    raise NotImplementedError("RAE lesson1.grid_to_tokens TODO 1.3")
