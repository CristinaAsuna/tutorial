"""第3关：EMA teacher 与 Gram teacher 的角色和更新频率不同。
前置：DINO EMA。输入completed_steps为已完成optimizer次数，非epoch/环境时间。
手算：snapshot=2,start=4,interval=2；2->snapshot，4->None，6->refresh。
接入：optimizer -> EMA -> model.after_step -> 本关，下一前向使用新快照。
Gram teacher为独立、冻结、eval backbone；不是每步EMA的另一个名字。
常见错误：两个teacher引用同一对象；先刷新后EMA；一开始即refinement。
检查：python3 -m mllm.dinov3.check_lessons --lesson 3
"""

def gram_event(completed_steps, snapshot_step=2, refine_start=4, refresh_every=2):
    # TODO 3.1: snapshot_step时返回'snapshot'。
    # TODO 3.2: 大于refine_start且间隔整除返回'refresh'；其余None。
    raise NotImplementedError('lesson3.gram_event TODO 3.1-3.2')


def sync_gram_teacher(gram_teacher, ema_backbone):
    # TODO 3.3: no_grad下load_state_dict，冻结Gram参数并eval；不改EMA。
    raise NotImplementedError('lesson3.sync_gram_teacher TODO 3.3')
