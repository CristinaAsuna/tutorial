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


class TubeletEmbed(nn.Module):
    """Turn a fixed-grid video into one token per spatiotemporal tubelet.

    The returned grid is ordered ``(time, height, width)`` and matches the
    flattened token order produced by ``Conv3d``: time changes slowest and
    width changes fastest.  Keeping that convention here prevents a video
    tutorial from having to duplicate fragile patch-index arithmetic.
    """

    def __init__(self, in_chans: int, embed_dim: int, tubelet_size: int, patch_size: int):
        super().__init__()
        if tubelet_size <= 0 or patch_size <= 0:
            raise ValueError("tubelet_size and patch_size must be positive")
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size
        kernel_size = (tubelet_size, patch_size, patch_size)
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size, kernel_size)

    def forward(self, videos: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        if videos.ndim != 5:
            raise ValueError("videos must have shape (B, C, T, H, W)")
        _, _, frames, height, width = videos.shape
        if frames % self.tubelet_size or height % self.patch_size or width % self.patch_size:
            raise ValueError("video T/H/W must be divisible by tubelet_size/patch_size")
        x = self.proj(videos)
        grid = tuple(x.shape[-3:])
        return x.flatten(2).transpose(1, 2), grid


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


def interpolate_3d_pos_embed(
    position_embeddings: torch.Tensor,
    old_grid: tuple[int, int, int],
    grid: tuple[int, int, int],
    *,
    num_prefix_tokens: int = 0,
) -> torch.Tensor:
    """Trilinearly resize learned video position embeddings, preserving prefixes."""
    if position_embeddings.ndim != 3 or position_embeddings.shape[0] != 1:
        raise ValueError("position_embeddings must have shape (1, N, D)")
    if any(size <= 0 for size in (*old_grid, *grid)):
        raise ValueError("old_grid and grid dimensions must be positive")
    prefix, patches = position_embeddings[:, :num_prefix_tokens], position_embeddings[:, num_prefix_tokens:]
    if patches.shape[1] != old_grid[0] * old_grid[1] * old_grid[2]:
        raise ValueError("patch position count must equal product of old_grid")
    patches = patches.reshape(1, *old_grid, -1).permute(0, 4, 1, 2, 3)
    patches = F.interpolate(patches, size=grid, mode="trilinear", align_corners=False)
    patches = patches.permute(0, 2, 3, 4, 1).reshape(1, grid[0] * grid[1] * grid[2], -1)
    return torch.cat((prefix, patches), dim=1)


def sincos_3d_pos_embed(grid: tuple[int, int, int], embed_dim: int, *, device=None, dtype=None) -> torch.Tensor:
    """Return deterministic ``(1, T*H*W, D)`` sin/cos positions for video tokens.

    Each axis receives an equal, even-dimensional chunk.  This intentionally
    explicit constraint avoids silently dropping channels in small toy models.
    """
    if len(grid) != 3 or any(size <= 0 for size in grid):
        raise ValueError("grid must be a three-tuple of positive (T, H, W) sizes")
    if embed_dim <= 0 or embed_dim % 6:
        raise ValueError("embed_dim must be a positive multiple of 6 for 3D sin/cos positions")
    axis_dim = embed_dim // 3
    omega = torch.arange(axis_dim // 2, device=device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (axis_dim // 2)))
    coords = torch.meshgrid(
        *(torch.arange(size, device=device, dtype=torch.float32) for size in grid), indexing="ij"
    )
    pieces = []
    for coord in coords:
        phase = coord.reshape(-1, 1) * omega
        pieces.append(torch.cat((phase.sin(), phase.cos()), dim=1))
    return torch.cat(pieces, dim=1).unsqueeze(0).to(dtype=dtype)
