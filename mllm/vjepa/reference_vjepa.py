"""小型、可读的原始 V-JEPA 教学实现。

它预测被遮蔽时空 tubelet 的 *teacher latent*，不是重建 RGB 像素。
本文件刻意使用小 VideoViT 和固定长度视频，便于在 CPU 上看清数据流；它
不是 Meta 的大规模 V-JEPA 训练系统。
"""
from __future__ import annotations

import copy
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from edu_core.training import freeze_and_keep_eval, update_ema
from edu_core.vision import TubeletEmbed


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 2.0):
        super().__init__()
        if dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
                                 nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, x: Tensor) -> Tensor:
        y = self.norm1(x)
        x = x + self.attn(y, y, y, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class MiniVideoViT(nn.Module):
    """固定基础网格的 VideoViT；context 路径可只接收可见 token。"""
    def __init__(self, video_size: tuple[int, int, int] = (8, 32, 32),
                 tubelet_size: int = 2, patch_size: int = 8, embed_dim: int = 64,
                 depth: int = 2, num_heads: int = 4, in_chans: int = 3):
        super().__init__()
        t, h, w = video_size
        if t % tubelet_size or h % patch_size or w % patch_size:
            raise ValueError("video_size must be divisible by tubelet_size/patch_size")
        self.tubelet_embed = TubeletEmbed(in_chans, embed_dim, tubelet_size, patch_size)
        self.base_grid = (t // tubelet_size, h // patch_size, w // patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, math.prod(self.base_grid), embed_dim))
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def tokens(self, videos: Tensor) -> tuple[Tensor, tuple[int, int, int]]:
        patches, grid = self.tubelet_embed(videos)
        if grid != self.base_grid:
            raise ValueError(f"expected tubelet grid {self.base_grid}, got {grid}; this toy model uses fixed video size")
        return patches + self.pos_embed, grid

    def forward_tokens(self, tokens: Tensor) -> Tensor:
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)

    def forward(self, videos: Tensor) -> Tensor:
        return self.forward_tokens(self.tokens(videos)[0])


def sample_spatiotemporal_masks(batch_size: int, grid: tuple[int, int, int],
                                target_block: tuple[int, int, int] = (1, 2, 2),
                                num_targets: int = 2, *, generator: torch.Generator | None = None,
                                device: torch.device | None = None) -> tuple[Tensor, Tensor]:
    """采样不重叠 target cuboids，context 恰为其补集。

    展平索引遵循 Conv3d 输出的 ``time, height, width`` 行主序。
    """
    if batch_size <= 0 or num_targets <= 0:
        raise ValueError("batch_size and num_targets must be positive")
    if len(grid) != 3 or len(target_block) != 3 or min(*grid, *target_block) <= 0:
        raise ValueError("grid and target_block must be positive 3-tuples")
    if any(a > b for a, b in zip(target_block, grid)):
        raise ValueError("target_block must fit inside grid")
    n = math.prod(grid)
    target = torch.zeros(batch_size, n, dtype=torch.bool, device=device)
    gt, gh, gw = grid
    bt, bh, bw = target_block
    for row in range(batch_size):
        occupied = torch.zeros(grid, dtype=torch.bool)
        for _ in range(num_targets):
            for _attempt in range(100):
                start_t = int(torch.randint(gt - bt + 1, (1,), generator=generator).item())
                start_h = int(torch.randint(gh - bh + 1, (1,), generator=generator).item())
                start_w = int(torch.randint(gw - bw + 1, (1,), generator=generator).item())
                area = occupied[start_t:start_t + bt, start_h:start_h + bh, start_w:start_w + bw]
                if not area.any():
                    area.fill_(True)
                    break
            else:
                raise ValueError("could not sample non-overlapping target blocks; reduce num_targets or block size")
        target[row] = occupied.reshape(-1).to(device=device)
    context = ~target
    return target, context


class LatentPredictor(nn.Module):
    """以可见 context token 与 target 位置，预测 target 表征。"""
    def __init__(self, embed_dim: int, predictor_dim: int = 48, depth: int = 2, num_heads: int = 4):
        super().__init__()
        self.context_proj = nn.Linear(embed_dim, predictor_dim)
        self.target_pos_proj = nn.Linear(embed_dim, predictor_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        self.blocks = nn.ModuleList([TransformerBlock(predictor_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(predictor_dim)
        self.out = nn.Linear(predictor_dim, embed_dim)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, context: Tensor, target_positions: Tensor) -> Tensor:
        if context.ndim != 3 or target_positions.ndim != 3 or context.shape[0] != target_positions.shape[0]:
            raise ValueError("context and target_positions must be [batch, tokens, dim] with equal batch size")
        visible = self.context_proj(context)
        target = self.mask_token.expand(target_positions.shape[0], target_positions.shape[1], -1)
        target = target + self.target_pos_proj(target_positions)
        x = torch.cat((visible, target), dim=1)
        for block in self.blocks:
            x = block(x)
        return self.out(self.norm(x[:, -target.shape[1]:]))


class VJEPA(nn.Module):
    """EMA target VideoViT + context VideoViT + target-latent predictor."""
    def __init__(self, context_encoder: MiniVideoViT, predictor: LatentPredictor):
        super().__init__()
        self.context_encoder = context_encoder
        self.target_encoder = copy.deepcopy(context_encoder)
        self.predictor = predictor
        self._freeze_target()

    def _freeze_target(self) -> None:
        freeze_and_keep_eval(self.target_encoder)

    def train(self, mode: bool = True) -> "VJEPA":
        super().train(mode)
        self._freeze_target()  # model.train() must never turn the EMA teacher back to train.
        return self

    @staticmethod
    def _validate_masks(target_mask: Tensor, context_mask: Tensor, batch: int, tokens: int) -> None:
        if target_mask.dtype != torch.bool or context_mask.dtype != torch.bool:
            raise ValueError("target_mask and context_mask must be boolean")
        if target_mask.shape != (batch, tokens) or context_mask.shape != (batch, tokens):
            raise ValueError(f"masks must have shape [{batch}, {tokens}]")
        if not target_mask.any(dim=1).all() or not context_mask.any(dim=1).all():
            raise ValueError("every video needs at least one target and one context tubelet")
        if not torch.equal(context_mask, ~target_mask):
            raise ValueError("context_mask must be exactly the complement of target_mask (no target/context leakage)")
        if len(torch.unique(target_mask.sum(1))) != 1:
            raise ValueError("all batch elements must have the same number of target tubelets")

    @staticmethod
    def _select(tokens: Tensor, mask: Tensor) -> Tensor:
        b, _, d = tokens.shape
        return tokens[mask].reshape(b, -1, d)

    def forward(self, videos: Tensor, target_mask: Tensor, context_mask: Tensor) -> dict[str, Tensor]:
        input_tokens, _ = self.context_encoder.tokens(videos)
        b, n, _ = input_tokens.shape
        self._validate_masks(target_mask, context_mask, b, n)
        # The teacher receives the complete video, but its target features are detached.
        with torch.no_grad():
            self.target_encoder.eval()
            target_features = self.target_encoder(videos)
        context_features = self.context_encoder.forward_tokens(self._select(input_tokens, context_mask))
        target_positions = self._select(self.context_encoder.pos_embed.expand(b, -1, -1), target_mask)
        predictions = self.predictor(context_features, target_positions)
        targets = self._select(target_features, target_mask)
        loss = F.smooth_l1_loss(predictions, targets)
        return {"loss": loss, "predictions": predictions, "targets": targets.detach(),
                "context_tokens": context_features, "target_mask": target_mask}

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("EMA momentum must be in [0, 1]")
        update_ema(self.target_encoder, self.context_encoder, momentum)
        self._freeze_target()
