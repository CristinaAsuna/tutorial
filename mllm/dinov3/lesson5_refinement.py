"""第5关：保留全局判别目标，加入局部关系约束。
前置：前四关与DINOv2 training。base为scalar，两个global的features为列表。
手算：base=2，两crop Gram loss=1和3，weight=.5 -> total=3。
接入：model.losses -> refinement_loss；demo -> training_step，均经过本关。
常见错误：以Gram替换全部原loss；teacher加入optimizer；更新后快照供当前步重用。
检查：python3 -m mllm.dinov3.check_lessons --lesson 5
"""

def refinement_loss(base, student_features, teacher_features, gram_fn, weight=1.):
    # TODO 5.1: 同crop配对调用gram_fn，对两个global平均。
    # TODO 5.2: 返回(base+weight*anchor,anchor)。
    raise NotImplementedError('lesson5.refinement_loss TODO 5.1-5.2')


def training_step(model, crops, masks, high_crops, optimizer, step, total_steps):
    # TODO 5.3: 验证step==model.completed_steps；train、清梯度。
    # TODO 5.4: model.losses(crops,masks,step,high_crops=high_crops)，backward与optimizer。
    # TODO 5.5: 更新EMA，再model.after_step(step+1)，返回detached损失字典。
    raise NotImplementedError('lesson5.training_step TODO 5.3-5.5')
