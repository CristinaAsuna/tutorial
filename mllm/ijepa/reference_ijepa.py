"""A small, explicit I-JEPA teaching implementation.

I-JEPA predicts *teacher representations* of hidden image blocks.  It does
not reconstruct pixels, use labels, negatives, or a DINO-style softmax head.
The code deliberately keeps one shared block mask for a mini-batch so that the
variable-length context sequence stays readable.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from edu_core.training import freeze_and_keep_eval, update_ema
from edu_core.vision import PatchEmbed


@dataclass(frozen=True)
class BlockMasks:
    """Target blocks and their complement on a ``grid_h * grid_w`` patch grid."""
    targets: tuple[Tensor, ...]  # each bool [num_patches]
    context: Tensor               # bool [num_patches], True means visible


def sample_block_masks(
    grid: tuple[int, int], *, num_targets: int = 2,
    block_size: tuple[int, int] = (2, 2),
    generator: torch.Generator | None = None,
) -> BlockMasks:
    """Sample non-overlapping rectangular target blocks, then take their complement."""
    gh, gw = grid
    bh, bw = block_size
    if gh <= 0 or gw <= 0 or bh <= 0 or bw <= 0 or bh > gh or bw > gw:
        raise ValueError("block_size must fit inside a positive patch grid")
    if num_targets <= 0 or num_targets * bh * bw > gh * gw:
        raise ValueError("num_targets must leave at least one context patch")
    placements = [(r, c) for r in range(gh - bh + 1) for c in range(gw - bw + 1)]
    # Backtracking makes sampling finite and prevents a valid request from
    # failing merely because an early random placement blocks later ones.
    order = torch.randperm(len(placements), generator=generator).tolist()

    def place(occupied: Tensor, chosen: list[Tensor]) -> list[Tensor] | None:
        if len(chosen) == num_targets:
            return chosen
        for choice in order:
            r, c = placements[choice]
            if occupied[r:r + bh, c:c + bw].any():
                continue
            block = torch.zeros_like(occupied)
            block[r:r + bh, c:c + bw] = True
            answer = place(occupied | block, chosen + [block.flatten()])
            if answer is not None:
                return answer
        return None

    blocks = place(torch.zeros(gh, gw, dtype=torch.bool), [])
    if blocks is None:
        raise ValueError("cannot place the requested non-overlapping target blocks")
    occupied = torch.stack(blocks).any(dim=0).reshape(gh, gw)
    context = ~occupied.flatten()
    if not context.any():
        raise ValueError("target blocks must leave at least one context patch")
    return BlockMasks(tuple(blocks), context)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 2.0):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, x: Tensor) -> Tensor:
        y = self.norm1(x)
        x = x + self.attn(y, y, y, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class TokenEncoder(nn.Module):
    """Patch embedding plus ViT encoder.  ``encode`` accepts a token subset."""
    def __init__(self, image_size: int = 32, patch_size: int = 8, dim: int = 64, depth: int = 2, heads: int = 4):
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.patch_embed = PatchEmbed(3, dim, patch_size)
        self.image_size, self.patch_size, self.dim = image_size, patch_size, dim
        n = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, n, dim))
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def patch_tokens(self, images: Tensor) -> tuple[Tensor, tuple[int, int]]:
        tokens, grid = self.patch_embed(images)
        if tokens.shape[1] != self.pos_embed.shape[1]:
            raise ValueError("this teaching encoder uses its configured image_size only")
        return tokens, grid

    def encode(self, patch_tokens: Tensor, indices: Tensor) -> Tensor:
        if indices.dtype != torch.long or indices.ndim != 1 or indices.numel() == 0:
            raise ValueError("indices must be a non-empty 1D long tensor")
        if indices.min() < 0 or indices.max() >= patch_tokens.shape[1]:
            raise ValueError("indices are outside the patch grid")
        x = patch_tokens.index_select(1, indices) + self.pos_embed.index_select(1, indices)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, images: Tensor) -> tuple[Tensor, tuple[int, int]]:
        patches, grid = self.patch_tokens(images)
        return self.encode(patches, torch.arange(patches.shape[1], device=patches.device)), grid


class JEPApredictor(nn.Module):
    """Predict hidden target slots after appending position-conditioned mask tokens."""
    def __init__(self, dim: int, predictor_dim: int = 96, depth: int = 2, heads: int = 4):
        super().__init__()
        self.in_proj = nn.Linear(dim, predictor_dim) if dim != predictor_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        self.blocks = nn.ModuleList([TransformerBlock(predictor_dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(predictor_dim)
        self.out_proj = nn.Linear(predictor_dim, dim)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, context: Tensor, target_pos: Tensor) -> Tensor:
        if context.ndim != 3 or target_pos.ndim != 3 or context.shape[0] != target_pos.shape[0]:
            raise ValueError("context and target_pos must be [batch, tokens, dim] with matching batch")
        context = self.in_proj(context)
        if target_pos.shape[-1] != context.shape[-1]:
            raise ValueError("target_pos must use predictor_dim")
        targets = self.mask_token.expand(context.shape[0], target_pos.shape[1], -1) + target_pos
        x = torch.cat((context, targets), dim=1)
        for block in self.blocks:
            x = block(x)
        return self.out_proj(self.norm(x[:, -target_pos.shape[1]:]))


class IJEPA(nn.Module):
    """Student context encoder + EMA target encoder + latent-space predictor."""
    def __init__(self, *, image_size: int = 32, patch_size: int = 8, dim: int = 64, depth: int = 2, heads: int = 4,
                 predictor_dim: int = 96, predictor_depth: int = 2):
        super().__init__()
        self.context_encoder = TokenEncoder(image_size, patch_size, dim, depth, heads)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self.predictor = JEPApredictor(dim, predictor_dim, predictor_depth, heads)
        self.target_pos_proj = nn.Linear(dim, predictor_dim) if dim != predictor_dim else nn.Identity()
        freeze_and_keep_eval(self.target_encoder)

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_encoder.eval()  # preserve the EMA teacher invariant
        return self

    def forward(self, images: Tensor, masks: BlockMasks) -> dict[str, Tensor]:
        patches, grid = self.context_encoder.patch_tokens(images)
        n = patches.shape[1]
        if masks.context.shape != (n,) or masks.context.dtype != torch.bool or not masks.targets:
            raise ValueError("masks must match the image patch count and contain targets")
        union = torch.zeros(n, dtype=torch.bool, device=patches.device)
        target_indices: list[Tensor] = []
        for target in masks.targets:
            if target.shape != (n,) or target.dtype != torch.bool:
                raise ValueError("each target mask must be bool [num_patches]")
            target = target.to(patches.device)
            if not target.any() or (union & target).any():
                raise ValueError("target blocks must be non-empty and non-overlapping")
            union |= target
            target_indices.append(target.nonzero(as_tuple=False).flatten())
        context_mask = masks.context.to(patches.device)
        if (context_mask & union).any() or not torch.equal(context_mask, ~union):
            raise ValueError("context must be exactly the complement of all target blocks")
        context_indices = context_mask.nonzero(as_tuple=False).flatten()
        all_target_indices = torch.cat(target_indices)
        context = self.context_encoder.encode(patches, context_indices)
        target_pos = self.target_pos_proj(self.context_encoder.pos_embed.index_select(1, all_target_indices)).expand(images.shape[0], -1, -1)
        predictions = self.predictor(context, target_pos)
        with torch.no_grad():
            self.target_encoder.eval()
            teacher_patches, teacher_grid = self.target_encoder.patch_tokens(images)
            if teacher_grid != grid:
                raise RuntimeError("student and teacher patch grids differ")
            targets = self.target_encoder.encode(teacher_patches, all_target_indices)
        loss = torch.nn.functional.smooth_l1_loss(predictions, targets)
        return {"loss": loss, "predictions": predictions, "targets": targets, "context_indices": context_indices, "target_indices": all_target_indices}

    @torch.no_grad()
    def update_target_encoder(self, momentum: float) -> None:
        update_ema(self.target_encoder, self.context_encoder, momentum)
        freeze_and_keep_eval(self.target_encoder)
