"""第7关：training。
前置：foundations patchify/attention/mask/EMA；各函数目的、形状和手算见下方。
接入：practice_ops把本关函数注入RAECourse；单关检查提供其他关参考外围。
检查（仓库根）：PYTHONPATH=general_utils/edu_core python3 -m mllm.rae.check_lessons --lesson 7
常见错误：对错轴归约、错时间方向、切断可训练梯度、把eval误当冻结。
"""
import torch
from torch.nn import functional as F
from edu_core.vision import unpatchify
from edu_core.training import update_ema

def stage2_step(course, images, optimizer, time=None, noise=None, momentum=.9):
    """目的：第二阶段冻结整个RAE，训练生成器；optimizer之后更新EMA。
    形状：images→latent(B,16,64)→shifted time→flow pair→predicted velocity→scalar。
    手算：EMA旧2、student更新后6、m=.75 → 新EMA=3。
    """
    # TODO 7.1.1: course.set_stage(2)，optimizer.zero_grad；latent=ops.grid_to_tokens(course.rae.encode_grid(images))。
    # TODO 7.1.2: 缺省time=rand(B)，通过ops.shift_time(time,N*D,course.reference_dim)偏移；缺省noise=randn_like(latent)。
    # TODO 7.1.3: 调用ops.flow_pair(latent,time,noise)，generator(noisy,time)预测速度，再调用ops.velocity_loss。
    # TODO 7.1.4: loss.backward()；optimizer.step()；随后update_ema(course.ema,course.generator,momentum)。
    # TODO 7.1.5: 返回loss.detach()，不能更新RAE参数或让EMA训练态。
    raise NotImplementedError("RAE lesson7.stage2_step TODO 7.1")


