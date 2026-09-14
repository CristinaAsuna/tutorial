"""Lesson 3: implement a stop-gradient EMA target encoder update."""
from torch import nn


def update_target(teacher: nn.Module, student: nn.Module, momentum: float) -> None:
    # TODO: EMA-update teacher parameters under no_grad, then keep it frozen/eval.
    raise NotImplementedError("Implement JEPA EMA teacher update")
