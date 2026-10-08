"""关卡 3：context 和目标位置预测 latent

前置：I-JEPA packing、共享 Transformer。
形状合同：context [B,C,D],pos [B,M,D] → prediction [B,M,D]。
手算例子：C=3,M=2：拼接长5，输出只取最后2个。
编号 TODO：1. 建立投影和 mask token。2. context 投影。3. target position 投影+mask。4. 拼接/blocks。5. norm/取最后M/输出投影。
常见错误：predictor_dim 可不同于 D；不可把完整视频 token 交给 student。
检查：python3 check_lessons.py --lesson 3 --implementation practice
提示：HINTS.md 第 3 关；参考检查可加 --implementation reference。
"""
import torch
from torch import nn


class Predictor(nn.Module):
    def __init__(self, dim: int, predictor_dim: int = 48, depth: int = 2, num_heads: int = 4):
        super().__init__()
        from reference_vjepa import TransformerBlock
        self.context_proj = nn.Linear(dim, predictor_dim)
        self.target_pos_proj = nn.Linear(dim, predictor_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        self.blocks = nn.ModuleList([TransformerBlock(predictor_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(predictor_dim)
        self.out = nn.Linear(predictor_dim, dim)
        # TODO 在 forward 按 2-5 步连接已提供的参数。

    def forward(self, context: torch.Tensor, target_positions: torch.Tensor) -> torch.Tensor:
        # TODO: 拼接 context 与 [mask token + target position] 后预测最后 M 个 token。
        x = self.pack_tokens(context, target_positions)
        for block in self.blocks:
            x = block(x)
        return self.select_predictions(x, target_positions.shape[1])

    def pack_tokens(self, context: torch.Tensor, target_positions: torch.Tensor) -> torch.Tensor:
        """微任务 1：两条投影 + target mask token，沿 token 轴拼接。"""
        # TODO 1: context_proj(context)，mask.expand + target_pos_proj(pos)，cat dim=1。
        raise NotImplementedError("lesson3.pack_tokens TODO 1")

    def select_predictions(self, tokens: torch.Tensor, num_targets: int) -> torch.Tensor:
        """微任务 2：仅取最后 M slots，norm 后 out projection，恢复 encoder D。"""
        # TODO 2: 不返回 context slots；最终 [B,M,D]。
        raise NotImplementedError("lesson3.select_predictions TODO 2")
