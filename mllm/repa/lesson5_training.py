"""第5关：更新student、projector与EMA，保持teacher/codec冻结。
训练返回三个detach标量；EMA同shape参数逐元素更新。
手算ema=2,student=6,m=.75 -> 3，m=1不变。
顺序train -> zero_grad -> system.forward -> backward -> step -> EMA。
前置foundations6；EMA只更新parameters，不代替external teacher。
zero-init下第一步部分gate/attention梯度为零正常，检查四步后的实际更新。
检查：python3 -m mllm.repa.check_lessons --lesson 5。
"""
import torch
from edu_core.training import update_ema


def ema_update(ema, model, momentum):
    # TODO 5.1：调用共享update_ema，momentum为旧EMA的权重。
    raise NotImplementedError('REPA lesson5.ema_update TODO 5.1')


def training_step(system, optimizer, images, labels, *, noise=None, time=None):
    # TODO 5.2：system.train()，清梯度，forward传入可选noise/time。
    # TODO 5.3：result['loss'].backward()，optimizer.step()。
    # TODO 5.4：system.ops.ema_update(ema_denoiser,denoiser,config.ema_momentum)。
    # TODO 5.5：返回每个值detach后的dict。
    raise NotImplementedError('REPA lesson5.training_step TODO 5.2-5.5')
