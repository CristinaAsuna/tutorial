"""第6关：不需要teacher/projector的Euler生成。
输入initial_noise=(B,4,8,8)，labels=(B,)，采样返回同shape；decode输出(B,3,16,16)。
公式x_next=x+(t_next-t)*v，t从1到0。
手算x=0,v=-2、steps=2，每步dt=-.5：0 -> 1 -> 2。
前置flow2；常见错：正步长、改变调用方noise、采样仍计算对齐分支。
system.sample外围已选择EMA/eval并no_grad，不需要干净图像。
检查：python3 -m mllm.repa.check_lessons --lesson 6。
"""
import torch


@torch.no_grad()
def euler_sample(model, labels, initial_noise, steps):
    # TODO 6.1：拒绝非正整数steps，clone初始noise。
    # TODO 6.2：第i步t=1-i/steps，构造(B,)同device/dtype时间。
    # TODO 6.3：model(...,project=False)只取velocity，current -= velocity/steps。
    raise NotImplementedError('REPA lesson6.euler_sample TODO 6.1-6.3')


def decode_latents(codec, latents):
    # TODO 6.4：返回codec.decode(latents)，不要在这里调用teacher。
    raise NotImplementedError('REPA lesson6.decode_latents TODO 6.4')
