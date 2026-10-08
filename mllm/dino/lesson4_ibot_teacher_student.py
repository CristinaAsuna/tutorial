"""关卡 4：masked patch 蒸馏与 teacher EMA

前置：上一关 soft CE、共享 EMA。
形状合同：patch [B,N,K], mask [B,N], center [1,K]。
手算例子：teacher=[1,0], student=[0.5,0.5] 时 CE=log(2)；t=2,s=6,m=.75 更新后 t=3。
编号 TODO：1. teacher detach/中心化。2. 每 patch CE。3. 仅 mask True 平均。4. no_grad EMA。
常见错误：分母是 masked patch 数量；optimizer 后才 EMA。
检查：python3 check_lessons.py --lesson 4 --implementation practice
提示：HINTS.md 第 4 关；参考检查可加 --implementation reference。
"""
from torch import Tensor


def ibot_masked_patch_loss(student_patch: Tensor, teacher_patch: Tensor, mask: Tensor,
                           patch_center: Tensor, student_temp: float, teacher_temp: float) -> Tensor:
    """Cross entropy only at mask == True positions; teacher targets are detached."""
    # TODO 1: teacher_patch.detach()，减 patch_center [1,K] 后温度 softmax。
    # TODO 2: student 温度 log_softmax；类别轴 sum 得 [B,N]。
    # TODO 3: per_patch[mask] 后 mean；未 mask 的 logits 不进入 loss。
    raise NotImplementedError("lesson4.ibot_masked_patch_loss TODO 1-3")


def ema_update(teacher_parameters, student_parameters, momentum: float) -> None:
    # TODO 4: 校验 momentum；torch.no_grad() 中 zip 同结构参数。
    # TODO 5: teacher.mul_(m).add_(student.detach(),alpha=1-m)，不更新 student。
    raise NotImplementedError("lesson4.ema_update TODO 4-5")
