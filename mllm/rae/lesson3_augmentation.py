"""第3关：augmentation。
前置：foundations patchify/attention/mask/EMA；各函数目的、形状和手算见下方。
接入：practice_ops把本关函数注入RAECourse；单关检查提供其他关参考外围。
检查（仓库根）：PYTHONPATH=general_utils/edu_core python3 -m mllm.rae.check_lessons --lesson 3
常见错误：对错轴归约、错时间方向、切断可训练梯度、把eval误当冻结。
"""
import torch
from torch.nn import functional as F
from edu_core.vision import unpatchify
from edu_core.training import update_ema

def augment_latent(latent, tau=.8, amplitude_normal=None, epsilon=None):
    """目的：为decoder训练增加离散latent附近的扰动，提高对生成误差的适应。
    形状：latent/epsilon=(B,N,D)，amplitude_normal=(B,)，sigma=(B,1,1)。
    手算：z=2,tau=.8,normal=-2,epsilon=3 → 3.2；sigma=.4可大于tau。
    """
    # TODO 3.1.1: 验证tau>=0；缺省amplitude_normal为randn(B)，epsilon为randn_like(latent)。
    # TODO 3.1.2: 验证随机输入shape；sigma=tau*abs(amplitude_normal)[:,None,None]。
    # TODO 3.1.3: 返回latent+sigma*epsilon；不要在这里做LN或改变噪声epsilon的分布。
    raise NotImplementedError("RAE lesson3.augment_latent TODO 3.1")


