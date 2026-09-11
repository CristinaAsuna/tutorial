"""Lesson 2: generate two global views, local views, and boolean patch masks."""
from torch import Tensor


def make_multicrop_views(images: Tensor) -> list[Tensor]:
    """Return [global_0, global_1, local_0, ..., local_3]."""
    raise NotImplementedError("Use MultiCropAugmentation in multicrop.py as the reference")


def random_patch_mask(batch_size: int, num_patches: int, ratio: float, device) -> Tensor:
    """Return bool [B, N], ensuring every image has at least one True entry."""
    raise NotImplementedError("Use make_patch_masks in multicrop.py as the reference")
