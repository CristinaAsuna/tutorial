"""ViT token primitives without model-specific objectives or augmentation policy."""
import torch
from torch import nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, patch_size: int):
        super().__init__()
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, patch_size)

    def forward(self, images: torch.Tensor):
        if images.ndim != 4 or images.shape[-2] % self.patch_size or images.shape[-1] % self.patch_size:
            raise ValueError("images must be BCHW with height/width divisible by patch_size")
        x = self.proj(images)
        return x.flatten(2).transpose(1, 2), tuple(x.shape[-2:])


def interpolate_2d_pos_embed(position_embeddings: torch.Tensor, grid: tuple[int, int], *, num_prefix_tokens: int = 1) -> torch.Tensor:
    """Bicubically resize a learned square patch grid while retaining prefix tokens."""
    if position_embeddings.ndim != 3 or position_embeddings.shape[0] != 1:
        raise ValueError("position_embeddings must have shape (1, N, D)")
    prefix, patches = position_embeddings[:, :num_prefix_tokens], position_embeddings[:, num_prefix_tokens:]
    old = int(patches.shape[1] ** 0.5)
    if old * old != patches.shape[1]:
        raise ValueError("patch position count must form a square grid")
    patches = patches.reshape(1, old, old, -1).permute(0, 3, 1, 2)
    patches = F.interpolate(patches, size=grid, mode="bicubic", align_corners=False)
    return torch.cat((prefix, patches.permute(0, 2, 3, 1).reshape(1, grid[0] * grid[1], -1)), dim=1)
