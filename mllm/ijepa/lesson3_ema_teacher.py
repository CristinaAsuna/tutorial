"""关卡 3：更新冻结的 target encoder

前置：共享 EMA 基础关。
形状合同：teacher/student 相同参数结构，返回 None。
手算例子：t=2,s=6,m=.75 → 3；m=1 保持 teacher。
编号 TODO：1. 校验 momentum。2. no_grad EMA。3. requires_grad False。4. eval。
常见错误：eval 不等于无梯度；不要反向更新 student。
检查：python3 check_lessons.py --lesson 3 --implementation practice
提示：HINTS.md 第 3 关；参考检查可加 --implementation reference。
"""
from torch import nn


def update_target(teacher: nn.Module, student: nn.Module, momentum: float) -> None:
    # TODO 1: 可调用基础关 edu_core.training.update_ema(teacher,student,momentum)。
    # TODO 2: freeze_and_keep_eval(teacher)，返回 None；eval 不代替 requires_grad=False。
    raise NotImplementedError("Implement JEPA EMA teacher update")
