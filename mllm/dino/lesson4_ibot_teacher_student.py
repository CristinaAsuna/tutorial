"""Lesson 4: add masked patch prediction and an EMA teacher."""
from torch import Tensor


def ibot_masked_patch_loss(student_patch: Tensor, teacher_patch: Tensor, mask: Tensor,
                           patch_center: Tensor, student_temp: float, teacher_temp: float) -> Tensor:
    """Cross entropy only at mask == True positions; teacher targets are detached."""
    raise NotImplementedError("See DINOiBOTLoss.forward in reference_dinov2.py")


def ema_update(teacher_parameters, student_parameters, momentum: float) -> None:
    raise NotImplementedError("Update after optimizer.step(): t = m*t + (1-m)*s")
