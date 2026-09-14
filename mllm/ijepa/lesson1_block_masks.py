"""Lesson 1: sample non-overlapping rectangular I-JEPA target masks."""
import torch


def sample_block_masks(grid: tuple[int, int], num_targets: int, block_size: tuple[int, int]) -> tuple[list[torch.Tensor], torch.Tensor]:
    # TODO: return target bool masks and their exact context complement.
    raise NotImplementedError("Implement 2D non-overlapping JEPA target blocks")
