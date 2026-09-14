"""V-JEPA 关卡 3：用可见 context 与 target 位置预测 target latent。"""
import torch
from torch import nn


class Predictor(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # TODO: 定义 mask token、位置投影、Transformer block 与输出头。
        raise NotImplementedError("TODO: 构造 latent predictor")

    def forward(self, context: torch.Tensor, target_positions: torch.Tensor) -> torch.Tensor:
        # TODO: 拼接 context 与 [mask token + target position] 后预测最后 M 个 token。
        raise NotImplementedError("TODO: 实现 predictor forward")
