"""关卡 3：跨视图软交叉熵与 center

前置：softmax/log_softmax、detach、EMA。
形状合同：student V×[B,K], teacher 2×[B,K], center [1,K]。
手算例子：2 global+1 local：有效配对 (t0,s1),(t0,s2),(t1,s0),(t1,s2) 共4；均匀 K=2 loss=log(2)。
编号 TODO：1. teacher 减 center/温度/detach。2. student log_softmax。3. 排除同 global。4. 平均配对。5. 更新 center。
常见错误：只跳过两个同 global 对；center 取 logits 均值而非概率。
检查：python3 check_lessons.py --lesson 3 --implementation practice
提示：HINTS.md 第 3 关；参考检查可加 --implementation reference。
"""
from torch import Tensor


def teacher_distribution(logits: Tensor, center: Tensor, temperature: float) -> Tensor:
    """微任务 1：[B,K] logits 与 [1,K] center → detached soft distribution [B,K]。"""
    # TODO 1a: detach；1b: (logits-center)/temperature；1c: softmax(dim=-1)。
    raise NotImplementedError("lesson3.teacher_distribution TODO 1")


def soft_cross_entropy(student: Tensor, target: Tensor, temperature: float) -> Tensor:
    """微任务 2：[B,K] → scalar；target 已 detach。均匀 K=2 得 log(2)。"""
    # TODO 2a: student/temperature 的 log_softmax。
    # TODO 2b: -(target*logp).sum(-1)；2c: 对 batch mean。
    raise NotImplementedError("lesson3.soft_cross_entropy TODO 2")


def dino_cross_view_loss(student_cls: list[Tensor], teacher_cls: list[Tensor], center: Tensor,
                         student_temp: float, teacher_temp: float) -> Tensor:
    """给出路由骨架；globals 必须在 student 列表前两项。"""
    if len(teacher_cls)!=2 or len(student_cls)<2:raise ValueError("need two teacher globals and at least two student globals")
    total=student_cls[0].new_zeros(());pairs=0
    for i,logits in enumerate(teacher_cls):
        target=teacher_distribution(logits,center,teacher_temp)
        for j,student in enumerate(student_cls):
            if i==j:continue
            total=total+soft_cross_entropy(student,target,student_temp)
            pairs+=1
    return total/pairs


def update_center(center: Tensor, teacher_logits: list[Tensor], momentum: float) -> Tensor:
    """微任务 3：[1,K] 与多个 [B,K] logits → 新 center [1,K]。"""
    # TODO 3a: 在无梯度上下文 cat 所有 teacher logits 并 mean(0,keepdim=True)。
    # TODO 3b: momentum*center + (1-momentum)*mean，返回新 buffer 值。
    raise NotImplementedError("lesson3.update_center TODO 3")
