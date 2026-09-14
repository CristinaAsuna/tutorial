"""Lesson 4: connect I-JEPA latent loss to an optimizer and frozen linear probe."""
from torch import Tensor


def jepa_training_step(model, images: Tensor, masks, optimizer):
    # TODO: forward, backward, optimizer step, target EMA update; return scalar loss.
    raise NotImplementedError("Implement an I-JEPA training step")
