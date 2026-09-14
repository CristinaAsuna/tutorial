"""Lesson 2: concatenate context representations with target mask-token slots."""
import torch


def pack_predictor_tokens(context: torch.Tensor, mask_token: torch.Tensor, target_positions: torch.Tensor) -> torch.Tensor:
    # TODO: make [B, N_context + N_target, D], with target slots = mask_token + position.
    raise NotImplementedError("Pack JEPA context and target slots")
