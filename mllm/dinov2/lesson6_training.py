"""第6关：把多个目标接成更新顺序明确的训练步骤。
前置：前五关，EMA基础；model.losses已连接各lesson，返回loss/dino/ibot/koleo。
形状：6个BCHW crop、2个B,N patch mask -> scalar -> 参数更新。
手算：warmup=4时step0/2/4温度 .04/.055/.07；EMA由.994升到1。
接入：run_dinov2_demo实际调用本关training_step。
常见错误：EMA早于optimizer；teacher加入optimizer；detach loss后再backward。
检查：python3 -m mllm.dinov2.check_lessons --lesson 6
"""

def teacher_temperature(step, warmup_steps=4, start=.04, end=.07):
    # TODO 6.1: 校验step与warmup非负；线性插值并在终点截断；warmup=0直接end。
    raise NotImplementedError('lesson6.teacher_temperature TODO 6.1')


def ema_momentum(step, total_steps, start=.994):
    # TODO 6.2: 可调用 edu_core.training.cosine_ema_momentum，end=1。
    raise NotImplementedError('lesson6.ema_momentum TODO 6.2')


def training_step(model, crops, masks, optimizer, step, total_steps):
    # TODO 6.3: train、zero_grad、model.losses、loss.backward、optimizer.step。
    # TODO 6.4: model.update_teacher(model.ops.ema_momentum(step,total_steps))。
    # TODO 6.5: 返回各项 detached loss，训练日志不保留计算图。
    raise NotImplementedError('lesson6.training_step TODO 6.3-6.5')
