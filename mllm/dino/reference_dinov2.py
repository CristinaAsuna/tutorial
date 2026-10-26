"""A deliberately small, readable DINO + iBOT teaching implementation.

This is not the production DINOv2 codebase.  It focuses on the self-distillation
loop: multi-crop student views, global teacher views, centering, EMA and masked
patch prediction.
"""
from __future__ import annotations

import copy
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from edu_core.training import freeze_and_keep_eval, update_ema


# Compatibility names; implementations live in the shared foundations package.
from edu_core.vision import PatchEmbed, interpolate_2d_pos_embed
from edu_core.attention import SelfAttentionBlock as AttentionBlock


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
        return interpolate_2d_pos_embed(self.pos_embed, grid)

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

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_teacher()
        return self

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
    def __init__(self, out_dim: int, student_temp: float = 0.1, teacher_temp: float = 0.07,
                 warmup_teacher_temp: float = 0.04, warmup_steps: int = 10,
                 center_momentum: float = 0.9, ibot_weight: float = 1.0):
        super().__init__()
        if student_temp <= 0 or teacher_temp <= 0 or warmup_teacher_temp <= 0:
            raise ValueError("temperatures must be positive")
        if warmup_steps < 0 or not 0 <= center_momentum <= 1 or ibot_weight < 0:
            raise ValueError("invalid warmup, center momentum or loss weight")
        self.student_temp, self.teacher_temp = student_temp, teacher_temp
        self.warmup_teacher_temp, self.warmup_steps = warmup_teacher_temp, warmup_steps
        self.center_momentum, self.ibot_weight = center_momentum, ibot_weight
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("patch_center", torch.zeros(1, out_dim))

    def current_teacher_temp(self, step: int) -> float:
        if step < 0:
            raise ValueError("step must be nonnegative")
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
