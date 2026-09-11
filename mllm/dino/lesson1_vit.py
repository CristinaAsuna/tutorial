"""Lesson 1: implement a minimal ViT.

Goal: turn BCHW images into patch tokens, add a CLS token and interpolate the
learned positional grid when a local crop has a different resolution.
"""
import torch
from torch import Tensor, nn


class ExerciseViT(nn.Module):
    def __init__(self, patch_size: int = 8, embed_dim: int = 48):
        super().__init__()
        self.patch_size, self.embed_dim = patch_size, embed_dim
        # TODO: add a Conv2d patch embedding, cls token, learned 2-D position grid,
        #       pre-LN transformer blocks and a final LayerNorm.

    def forward(self, images: Tensor, patch_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Return CLS [B, D] and patch tokens [B, N, D]."""
        raise NotImplementedError("Complete this lesson, then compare MiniViT in reference_dinov2.py")


if __name__ == "__main__":
    raise NotImplementedError("Exercise file: use reference_dinov2.py for the runnable answer")
