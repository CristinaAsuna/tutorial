"""Torch-only multi-crop augmentation and patch-mask helpers."""
from __future__ import annotations

from typing import Sequence
import torch
from torch import Tensor
import torch.nn.functional as F


def random_resized_crop(images: Tensor, output_size: int, scale: tuple[float, float]) -> Tensor:
    """Crop one random rectangle for the whole batch, then bilinearly resize it."""
    if images.ndim != 4 or not 0 < scale[0] <= scale[1] <= 1:
        raise ValueError("images must be BCHW and scale must satisfy 0 < min <= max <= 1")
    _, _, h, w = images.shape
    ratio = float(torch.empty((), device=images.device).uniform_(*scale))
    crop_h, crop_w = max(1, int(h * ratio)), max(1, int(w * ratio))
    top = int(torch.randint(h - crop_h + 1, (), device=images.device))
    left = int(torch.randint(w - crop_w + 1, (), device=images.device))
    return F.interpolate(images[:, :, top:top + crop_h, left:left + crop_w], size=(output_size, output_size), mode="bilinear", align_corners=False)


class MultiCropAugmentation:
    def __init__(self, global_size: int = 32, local_size: int = 16, num_local_crops: int = 4,
                 global_scale: tuple[float, float] = (0.5, 1.0), local_scale: tuple[float, float] = (0.2, 0.5)):
        self.global_size, self.local_size, self.num_local_crops = global_size, local_size, num_local_crops
        self.global_scale, self.local_scale = global_scale, local_scale

    def __call__(self, images: Tensor) -> list[Tensor]:
        globals_ = [random_resized_crop(images, self.global_size, self.global_scale) for _ in range(2)]
        locals_ = [random_resized_crop(images, self.local_size, self.local_scale) for _ in range(self.num_local_crops)]
        return globals_ + locals_


def make_patch_masks(global_crops: Sequence[Tensor], patch_size: int, mask_ratio: float = 0.5) -> list[Tensor]:
    if not 0 < mask_ratio <= 1:
        raise ValueError("mask_ratio must be in (0, 1]")
    masks = []
    for crop in global_crops:
        if crop.ndim != 4 or crop.shape[-1] % patch_size or crop.shape[-2] % patch_size:
            raise ValueError("global crops must be BCHW with dimensions divisible by patch_size")
        b = crop.shape[0]
        n = (crop.shape[-2] // patch_size) * (crop.shape[-1] // patch_size)
        count = max(1, int(round(n * mask_ratio)))
        noise = torch.rand(b, n, device=crop.device)
        mask = torch.zeros(b, n, dtype=torch.bool, device=crop.device)
        mask.scatter_(1, noise.argsort(dim=1)[:, :count], True)
        masks.append(mask)
    return masks
