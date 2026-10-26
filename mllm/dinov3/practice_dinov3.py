"""DINOv2 is a completed prerequisite; this course exercises the new Gram mechanism."""
from functools import partial
from . import reference_dinov3 as ref
from . import lesson1_gram as l1, lesson2_anchoring as l2, lesson3_teacher as l3
from . import lesson4_resolution as l4, lesson5_refinement as l5


def practice_ops():
    ops = ref.reference_ops()
    ops.gram_matrix=l1.gram_matrix
    ops.gram_loss=partial(l2.gram_loss,gram_fn=l1.gram_matrix)
    ops.gram_event=l3.gram_event
    ops.sync_gram_teacher=l3.sync_gram_teacher
    ops.aligned_crops=l4.aligned_crops
    ops.downsample_features=l4.downsample_features
    ops.refinement_loss=l5.refinement_loss
    ops.training_step=l5.training_step
    return ops


def build_toy_dinov3(**kwargs):
    return ref.DINOv3(ops=practice_ops(),**kwargs)
