"""Lesson 3: DINO cross-view loss with teacher centering and temperature."""
from torch import Tensor


def dino_cross_view_loss(student_cls: list[Tensor], teacher_cls: list[Tensor], center: Tensor,
                         student_temp: float, teacher_temp: float) -> Tensor:
    """Match every student crop to both teacher global crops, except same-view pairs."""
    raise NotImplementedError("See DINOiBOTLoss.forward in reference_dinov2.py")


def update_center(center: Tensor, teacher_logits: list[Tensor], momentum: float) -> Tensor:
    raise NotImplementedError("The center is an EMA of mean teacher logits")
