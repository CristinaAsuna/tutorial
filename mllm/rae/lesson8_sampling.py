"""第8关：sampling。
前置：foundations patchify/attention/mask/EMA；各函数目的、形状和手算见下方。
接入：practice_ops把本关函数注入RAECourse；单关检查提供其他关参考外围。
检查（仓库根）：PYTHONPATH=general_utils/edu_core python3 -m mllm.rae.check_lessons --lesson 8
常见错误：对错轴归约、错时间方向、切断可训练梯度、把eval误当冻结。
"""
import torch
from torch.nn import functional as F
from edu_core.vision import unpatchify
from edu_core.training import update_ema

def euler_sample(model, initial_noise, steps=8):
    """目的：从噪声时间1沿负步长积分到数据时间0，不再加入随机噪声。
    形状：initial_noise=(B,16,64) → 相同shape latent。
    手算：x=0,v=-2，偏移网格1→1/3→0，两步x: 0→4/3→2。
    """
    # TODO 8.1.1: 验证steps>0；在torch.no_grad()中clone initial_noise。
    # TODO 8.1.2: grid=model.ops.shift_time(linspace(1,0,steps+1),N*D,4096)。
    # TODO 8.1.3: 循环i：t=grid[i].expand(B)，delta=grid[i+1]-grid[i]。
    # TODO 8.1.4: x=x+delta*model(x,t)，循环后返回x；不得使用固定-1/steps或修改initial_noise。
    raise NotImplementedError("RAE lesson8.euler_sample TODO 8.1")


def sample_images(course, initial_noise, steps=8):
    """目的：完整生成：EMA速度场→latent→冻结decoder→图像。
    形状：noise=(B,16,64) → images=(B,3,16,16)。
    手算：16个4×4像素块拼成16×16图像，不把64通道latent直接当RGB。
    """
    # TODO 8.2.1: 在torch.no_grad()中course.set_stage(2)，保持RAE/EMA冻结eval。
    # TODO 8.2.2: 调用course.ops.euler_sample(course.ema,initial_noise,steps)得到latent。
    # TODO 8.2.3: return course.rae.decode(latent)，不要调用参考采样或直接将latent当RGB。
    raise NotImplementedError("RAE lesson8.sample_images TODO 8.2")


