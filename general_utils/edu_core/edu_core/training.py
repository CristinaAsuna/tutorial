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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
