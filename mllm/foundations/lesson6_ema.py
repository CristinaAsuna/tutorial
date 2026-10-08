"""基础 6：EMA 参数更新。

前置：requires_grad、no_grad、optimizer.step；EMA 不是反向传播。
公式：teacher = m*teacher + (1-m)*student；只更新 parameters，不更新 buffers。
例子：teacher=2，student=6，m=.75，更新后 teacher=3。
常见错：在 optimizer.step 前更新；使 teacher 带梯度；更新顺序错配。
接入：DINO/JEPA 学完后复用 edu_core.update_ema，各论文负责 frozen/eval 策略。
检查：python3 check_lessons.py --lesson 6；提示：HINTS.md 的基础 6。
"""
import torch


@torch.no_grad()
def update_ema(teacher, student, momentum):
    if not 0 <= momentum <= 1:
        raise ValueError("momentum 必须位于 [0,1]")
    # TODO 6.1：zip(teacher.parameters(), student.parameters(), strict=True)。
    # TODO 6.2：原地计算公式，不能重新绑定局部变量来替换 Parameter。
    raise NotImplementedError("基础 6.1–6.2：update_ema")
