"""Small training-state helpers; no trainer or data-loader policy is imposed."""
import random
import torch
from torch import nn


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(requires_grad)


def freeze_and_keep_eval(module: nn.Module) -> None:
    set_requires_grad(module, False)
    module.eval()


def update_ema(teacher: nn.Module, student: nn.Module, momentum: float) -> None:
    if not 0.0 <= momentum <= 1.0:
        raise ValueError("momentum must be in [0, 1]")
    with torch.no_grad():
        for target, source in zip(teacher.parameters(), student.parameters(), strict=True):
            target.mul_(momentum).add_(source, alpha=1.0 - momentum)


def cosine_ema_momentum(step: int, total_steps: int, *, start: float = 0.996, end: float = 1.0) -> float:
    """Cosine-ramp an EMA momentum from ``start`` to ``end`` inclusively."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0 <= step <= total_steps:
        raise ValueError("step must be in [0, total_steps]")
    if not 0.0 <= start <= end <= 1.0:
        raise ValueError("start and end must satisfy 0 <= start <= end <= 1")
    import math
    return end - (end - start) * (math.cos(math.pi * step / total_steps) + 1.0) / 2.0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
