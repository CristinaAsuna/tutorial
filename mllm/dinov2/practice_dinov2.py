"""Explicit composition. Completed earlier-course DINO/ViT is provided background."""
from functools import partial
from types import SimpleNamespace
from . import lesson1_masking as l1, lesson2_heads as l2, lesson3_sinkhorn as l3
from . import lesson4_ibot as l4, lesson5_koleo as l5, lesson6_training as l6
from . import reference_dinov2 as ref
from .scaffolding import build_modules


def practice_ops():
    return SimpleNamespace(
        sample_masks=partial(ref.sample_masks,rectangle_fn=l1.rectangle_mask,count_fn=l1.mask_count),
        build_modules=partial(build_modules,l2.choose_patch_head,l2.initialize_teacher),
        sinkhorn=l3.sinkhorn,dino_loss=ref.dino_loss,ibot_loss=l4.ibot_loss,koleo=l5.koleo,
        teacher_temperature=l6.teacher_temperature,ema_momentum=l6.ema_momentum,training_step=l6.training_step)


def build_toy_dinov2(**kwargs):
    return ref.DINOv2(ops=practice_ops(),**kwargs)
