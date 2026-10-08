"""关卡 1：图像变成 CLS 与 patch 表征

前置：PatchEmbed、attention 基础关。
形状合同：[B,3,H,W] → CLS [B,D], patches [B,N,D]。
手算例子：P=2,H=W=4 时 N=4；加 CLS 后序列长 5。
编号 TODO：1. 建立 patch/CLS/mask/position 参数。2. mask 替换。3. 插值位置。4. Transformer 和输出切片。
常见错误：CLS 不属于 patch mask；先插值 patch 网格再拼 CLS 位置。
检查：python3 check_lessons.py --lesson 1 --implementation practice
提示：HINTS.md 第 1 关；参考检查可加 --implementation reference。
"""
import torch
from torch import Tensor, nn


class ExerciseViT(nn.Module):
    def __init__(self, patch_size: int = 8, embed_dim: int = 48, image_size: int = 32, depth: int = 2, num_heads: int = 4, in_chans: int = 3):
        super().__init__()
        self.patch_size, self.embed_dim = patch_size, embed_dim
        from edu_core.vision import PatchEmbed
        from reference_dinov2 import AttentionBlock
        self.patch_embed = PatchEmbed(in_chans, embed_dim, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + (image_size // patch_size)**2, embed_dim))
        self.blocks = nn.ModuleList([AttentionBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        # TODO 1-4 在 forward 完成；参数骨架已提供，避免重复基础关。

    def forward(self, images: Tensor, patch_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Return CLS [B, D] and patch tokens [B, N, D]."""
        patches, grid = self.patch_embed(images)
        patches = self.replace_masked_patches(patches, patch_mask)
        x = self.add_cls_and_positions(patches, grid)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0], x[:, 1:]

    def replace_masked_patches(self, patches: Tensor, patch_mask: Tensor | None) -> Tensor:
        """微任务 1：bool [B,N] 广播成 [B,N,1]，用 mask_token 替换 True。"""
        # TODO 1: mask=None 时原样返回；检查 shape/dtype 后 torch.where。
        raise NotImplementedError("lesson1.replace_masked_patches TODO 1")

    def add_cls_and_positions(self, patches: Tensor, grid: tuple[int, int]) -> Tensor:
        """微任务 2：扩展 CLS、拼接，再加按 grid 插值的 positions。"""
        # TODO 2: 可以调用 edu_core.vision.interpolate_2d_pos_embed。
        raise NotImplementedError("lesson1.add_cls_and_positions TODO 2")


if __name__ == "__main__":
    raise NotImplementedError("Exercise file: use reference_dinov2.py for the runnable answer")
