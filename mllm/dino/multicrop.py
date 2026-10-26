"""Torch-only per-image area/aspect crop sampling. True patch_mask means hidden."""
from __future__ import annotations
import math
from typing import Sequence
import torch
from torch import Tensor
import torch.nn.functional as F


def crop_and_resize(images: Tensor, boxes: Tensor, output_size: int) -> Tensor:
    """Deterministic helper: boxes [B,4] are integer (top,left,height,width)."""
    if images.ndim != 4 or output_size <= 0 or boxes.shape != (images.shape[0], 4):
        raise ValueError("need BCHW, Bx4 boxes and positive output_size")
    if boxes.dtype not in (torch.int32, torch.int64):
        raise ValueError("boxes must be integer pixel coordinates")
    crops = []
    for image, box in zip(images, boxes):
        top, left, h, w = box.tolist()
        if min(top, left) < 0 or min(h, w) <= 0 or top+h > image.shape[-2] or left+w > image.shape[-1]:
            raise ValueError("crop box outside image")
        crops.append(F.interpolate(image[None, :, top:top+h, left:left+w], (output_size, output_size),
                                   mode="bilinear", align_corners=False))
    return torch.cat(crops)


def sample_crop_boxes(images: Tensor, scale: tuple[float, float], aspect=(.75, 4/3)) -> Tensor:
    """Sample each image independently. scale is area fraction, aspect is width/height.

    Ten candidate attempts, then a centered aspect-constrained fallback, as in
    RandomResizedCrop. Geometric augmentation only; no photometric transforms.
    """
    if images.ndim != 4 or not 0 < scale[0] <= scale[1] <= 1 or not 0 < aspect[0] <= aspect[1]:
        raise ValueError("invalid BCHW, area scale or aspect range")
    b, _, h, w = images.shape
    boxes = []
    for _ in range(b):
        for _ in range(10):
            area = h*w*float(torch.empty((), device=images.device).uniform_(*scale))
            ratio = math.exp(float(torch.empty((), device=images.device).uniform_(math.log(aspect[0]), math.log(aspect[1]))))
            ch, cw = round(math.sqrt(area/ratio)), round(math.sqrt(area*ratio))
            if 0 < ch <= h and 0 < cw <= w:
                top = int(torch.randint(h-ch+1, (), device=images.device))
                left = int(torch.randint(w-cw+1, (), device=images.device))
                break
        else:
            ratio = min(max(w/h, aspect[0]), aspect[1])
            cw, ch = (w, round(w/ratio)) if w/h < aspect[0] else (round(h*ratio), h)
            top, left = (h-ch)//2, (w-cw)//2
        boxes.append((top, left, ch, cw))
    return torch.tensor(boxes, device=images.device)


def random_resized_crop(images: Tensor, output_size: int, scale: tuple[float, float]) -> Tensor:
    return crop_and_resize(images, sample_crop_boxes(images, scale), output_size)


class MultiCropAugmentation:
    def __init__(self, global_size=32, local_size=16, num_local_crops=4,
                 global_scale=(.5, 1.), local_scale=(.2, .5)):
        self.global_size, self.local_size, self.num_local_crops = global_size, local_size, num_local_crops
        self.global_scale, self.local_scale = global_scale, local_scale

    def __call__(self, images: Tensor) -> list[Tensor]:
        return ([random_resized_crop(images, self.global_size, self.global_scale) for _ in range(2)] +
                [random_resized_crop(images, self.local_size, self.local_scale) for _ in range(self.num_local_crops)])


def make_patch_masks(global_crops: Sequence[Tensor], patch_size: int, mask_ratio: float = .5) -> list[Tensor]:
    if not 0 < mask_ratio <= 1 or patch_size <= 0:
        raise ValueError("mask_ratio in (0,1], patch_size positive")
    masks = []
    for crop in global_crops:
        if crop.ndim != 4 or crop.shape[-1] % patch_size or crop.shape[-2] % patch_size:
            raise ValueError("crops must be BCHW divisible by patch_size")
        b = crop.shape[0]
        n = (crop.shape[-2]//patch_size)*(crop.shape[-1]//patch_size)
        count = max(1, round(n*mask_ratio))
        order = torch.rand(b, n, device=crop.device).argsort(1)
        mask = torch.zeros(b, n, dtype=torch.bool, device=crop.device)
        mask.scatter_(1, order[:, :count], True)
        masks.append(mask)
    return masks
