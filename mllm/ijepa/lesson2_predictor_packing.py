"""关卡 2：拼接 context 和位置条件 target slots

前置：expand/cat 与共享 Transformer。
形状合同：context [B,C,D], mask [1,1,D], pos [B,M,D] → [B,C+M,D]。
手算例子：context=[1,2],mask=10,pos=[3,4] → [1,2,13,14]。
编号 TODO：1. 对齐 batch 和 D。2. expand mask。3. mask+target pos。4. dim=1 拼接。
常见错误：target slot 不能包含 teacher latent；不要沿 feature 轴拼。
检查：python3 check_lessons.py --lesson 2 --implementation practice
提示：HINTS.md 第 2 关；参考检查可加 --implementation reference。
"""
import torch


def pack_predictor_tokens(context: torch.Tensor, mask_token: torch.Tensor, target_positions: torch.Tensor) -> torch.Tensor:
    # TODO 1: context.shape 得 B,C,D；target_positions 得 B,M,D。
    # TODO 2: mask_token.expand(B,M,D)，加 target_positions 得 target slots。
    # TODO 3: torch.cat((context,target_slots),dim=1)，返回 [B,C+M,D]。
    raise NotImplementedError("Pack JEPA context and target slots")
