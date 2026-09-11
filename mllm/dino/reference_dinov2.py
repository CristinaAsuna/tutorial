"""A deliberately small, readable DINO + iBOT teaching implementation.

This is not the production DINOv2 codebase.  It focuses on the self-distillation
loop: multi-crop student views, global teacher views, centering, EMA and masked
patch prediction.
"""
from __future__ import annotations

import copy
import math
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from edu_core.training import freeze_and_keep_eval, update_ema


class PatchEmbed(nn.Module):
    def __init__(self, in_chans: int = 3, embed_dim: int = 192, patch_size: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, images: Tensor) -> tuple[Tensor, tuple[int, int]]:
        if images.ndim != 4:
            raise ValueError("images must have shape [batch, channels, height, width]")
        h, w = images.shape[-2:]
        if h % self.patch_size or w % self.patch_size:
            raise ValueError("image height and width must be divisible by patch_size")
        x = self.proj(images)
        grid = x.shape[-2:]
        return x.flatten(2).transpose(1, 2), grid


class AttentionBlock(nn.Module):
    """Pre-LayerNorm transformer block, written without hiding the attention."""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

    def forward(self, x: Tensor) -> Tensor:
        b, n, d = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(b, n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        attention = (q.transpose(1, 2) @ k.transpose(1, 2).transpose(-2, -1)) / math.sqrt(self.head_dim)
        x = x + self.proj((attention.softmax(dim=-1) @ v.transpose(1, 2)).transpose(1, 2).reshape(b, n, d))
        return x + self.mlp(self.norm2(x))


class MiniViT(nn.Module):
    """ViT with interpolated learned position embeddings and an optional mask token."""
    def __init__(self, image_size: int = 32, patch_size: int = 8, embed_dim: int = 96,
                 depth: int = 2, num_heads: int = 4, in_chans: int = 3):
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.patch_embed = PatchEmbed(in_chans, embed_dim, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        base_grid = image_size // patch_size
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + base_grid * base_grid, embed_dim))
        self.blocks = nn.ModuleList([AttentionBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def interpolated_pos_embed(self, grid: tuple[int, int]) -> Tensor:
        cls_pos, patch_pos = self.pos_embed[:, :1], self.pos_embed[:, 1:]
        old = int(math.sqrt(patch_pos.shape[1]))
        if old * old != patch_pos.shape[1]:
            raise RuntimeError("base position embedding must be a square grid")
        patch_pos = patch_pos.reshape(1, old, old, -1).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=grid, mode="bicubic", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, grid[0] * grid[1], -1)
        return torch.cat((cls_pos, patch_pos), dim=1)

    def forward(self, images: Tensor, patch_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        patches, grid = self.patch_embed(images)
        b, n, _ = patches.shape
        if patch_mask is not None:
            if patch_mask.shape != (b, n) or patch_mask.dtype != torch.bool:
                raise ValueError(f"patch_mask must be bool [batch, {n}], got {tuple(patch_mask.shape)}")
            patches = torch.where(patch_mask.unsqueeze(-1), self.mask_token.expand(b, n, -1), patches)
        x = torch.cat((self.cls_token.expand(b, -1, -1), patches), dim=1)
        x = x + self.interpolated_pos_embed(grid)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0], x[:, 1:]


class DINOHead(nn.Module):
    """The same prototype head is used for CLS tokens and patch tokens."""
    def __init__(self, in_dim: int, out_dim: int = 256, hidden_dim: int = 256, bottleneck_dim: int = 64):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, bottleneck_dim))
        self.prototypes = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.prototypes.weight_g.data.fill_(1.0)
        self.prototypes.weight_g.requires_grad = False

    def forward(self, tokens: Tensor) -> Tensor:
        return self.prototypes(F.normalize(self.mlp(tokens), dim=-1))

    def teacher_copy(self) -> "DINOHead":
        """Copy parameters without deepcopying legacy weight_norm tensors.

        PyTorch 2.13 rejects deepcopy of the non-leaf tensors created by
        ``nn.utils.weight_norm``.  Reconstructing from configuration gives the
        EMA teacher identical weights while remaining version-compatible.
        """
        teacher = DINOHead(self.in_dim, self.out_dim, self.hidden_dim, self.bottleneck_dim)
        teacher.load_state_dict(self.state_dict())
        return teacher


class TeacherStudentDINO(nn.Module):
    def __init__(self, student_backbone: MiniViT, student_head: DINOHead):
        super().__init__()
        self.student_backbone, self.student_head = student_backbone, student_head
        self.teacher_backbone = copy.deepcopy(student_backbone)
        self.teacher_head = student_head.teacher_copy()
        self._freeze_teacher()

    def _freeze_teacher(self) -> None:
        freeze_and_keep_eval(self.teacher_backbone)
        freeze_and_keep_eval(self.teacher_head)

    def teacher_parameters(self) -> Iterable[nn.Parameter]:
        return list(self.teacher_backbone.parameters()) + list(self.teacher_head.parameters())

    def student_forward(self, crops: Sequence[Tensor], masks: Sequence[Tensor | None]) -> list[dict[str, Tensor]]:
        if len(crops) != len(masks):
            raise ValueError("crops and masks must have equal length")
        outputs = []
        for crop, mask in zip(crops, masks):
            cls, patch = self.student_backbone(crop, mask)
            outputs.append({"cls": self.student_head(cls), "patch": self.student_head(patch)})
        return outputs

    @torch.no_grad()
    def teacher_forward(self, global_crops: Sequence[Tensor]) -> list[dict[str, Tensor]]:
        if len(global_crops) != 2:
            raise ValueError("teacher must receive exactly two global crops")
        self.teacher_backbone.eval()
        self.teacher_head.eval()
        result = []
        for crop in global_crops:
            cls, patch = self.teacher_backbone(crop)
            result.append({"cls": self.teacher_head(cls), "patch": self.teacher_head(patch)})
        return result

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("EMA momentum must be in [0, 1]")
        update_ema(self.teacher_backbone, self.student_backbone, momentum)
        update_ema(self.teacher_head, self.student_head, momentum)
        self._freeze_teacher()


class DINOiBOTLoss(nn.Module):
    def __init__(self, out_dim: int, student_temp: float = 0.1, teacher_temp: float = 0.04,
                 warmup_teacher_temp: float = 0.1, warmup_steps: int = 10,
                 center_momentum: float = 0.9, ibot_weight: float = 1.0):
        super().__init__()
        if student_temp <= 0 or teacher_temp <= 0 or warmup_teacher_temp <= 0:
            raise ValueError("temperatures must be positive")
        self.student_temp, self.teacher_temp = student_temp, teacher_temp
        self.warmup_teacher_temp, self.warmup_steps = warmup_teacher_temp, warmup_steps
        self.center_momentum, self.ibot_weight = center_momentum, ibot_weight
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("patch_center", torch.zeros(1, out_dim))

    def current_teacher_temp(self, step: int) -> float:
        if self.warmup_steps <= 0 or step >= self.warmup_steps:
            return self.teacher_temp
        alpha = step / self.warmup_steps
        return self.warmup_teacher_temp + alpha * (self.teacher_temp - self.warmup_teacher_temp)

    @torch.no_grad()
    def _update_center(self, teacher_outputs: Sequence[dict[str, Tensor]]) -> None:
        cls_mean = torch.cat([x["cls"].detach() for x in teacher_outputs]).mean(0, keepdim=True)
        patch_mean = torch.cat([x["patch"].detach().flatten(0, 1) for x in teacher_outputs]).mean(0, keepdim=True)
        self.center.mul_(self.center_momentum).add_(cls_mean, alpha=1 - self.center_momentum)
        self.patch_center.mul_(self.center_momentum).add_(patch_mean, alpha=1 - self.center_momentum)

    def forward(self, student_outputs: Sequence[dict[str, Tensor]], teacher_outputs: Sequence[dict[str, Tensor]],
                global_masks: Sequence[Tensor], step: int) -> dict[str, Tensor]:
        if len(teacher_outputs) != 2 or len(student_outputs) < 2 or len(global_masks) != 2:
            raise ValueError("need two teacher outputs, at least two student outputs, and two global masks")
        temp = self.current_teacher_temp(step)
        teacher_cls = [F.softmax((x["cls"].detach() - self.center) / temp, dim=-1) for x in teacher_outputs]
        dino_total = student_outputs[0]["cls"].new_zeros(())
        pairs = 0
        for teacher_index, target in enumerate(teacher_cls):
            for student_index, output in enumerate(student_outputs):
                if student_index == teacher_index:  # skip identical global view only
                    continue
                dino_total = dino_total + -(target * F.log_softmax(output["cls"] / self.student_temp, dim=-1)).sum(-1).mean()
                pairs += 1
        dino_loss = dino_total / pairs

        ibot_total = student_outputs[0]["cls"].new_zeros(())
        masked_count = 0
        for i, mask in enumerate(global_masks):
            student_patch, teacher_patch = student_outputs[i]["patch"], teacher_outputs[i]["patch"].detach()
            if mask.dtype != torch.bool or mask.shape != student_patch.shape[:2] or teacher_patch.shape != student_patch.shape:
                raise ValueError("each global mask must match its student/teacher patch logits [batch, patches]")
            if not mask.any():
                raise ValueError("each global crop must mask at least one patch")
            target = F.softmax((teacher_patch - self.patch_center) / temp, dim=-1)
            per_patch = -(target * F.log_softmax(student_patch / self.student_temp, dim=-1)).sum(-1)
            ibot_total = ibot_total + per_patch[mask].sum()
            masked_count += int(mask.sum())
        ibot_loss = ibot_total / masked_count
        total = dino_loss + self.ibot_weight * ibot_loss
        self._update_center(teacher_outputs)
        return {"loss": total, "dino_loss": dino_loss.detach(), "ibot_loss": ibot_loss.detach(), "teacher_temp": total.new_tensor(temp)}
