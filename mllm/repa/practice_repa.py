"""All paper computations route to student functions; no answer fallback."""
from types import SimpleNamespace
from . import lesson1_clean_targets as l1, lesson2_flow as l2, lesson3_hidden as l3
from . import lesson4_alignment as l4, lesson5_training as l5, lesson6_sampling as l6
from .scaffolding import RepaSystem, RepaConfig

LESSON_FUNCTIONS = {
    1: {'clean_targets':l1.clean_targets},
    2: {'flow_pair':l2.flow_pair},
    3: {'conditioned_block':l3.conditioned_block,'denoise':l3.denoise},
    4: {'alignment_loss':l4.alignment_loss,'joint_objective':l4.joint_objective},
    5: {'ema_update':l5.ema_update,'training_step':l5.training_step},
    6: {'euler_sample':l6.euler_sample,'decode_latents':l6.decode_latents},
}


def practice_ops():
    # Read current functions so test injection is independent of TODO completion.
    modules = [l1,l2,l3,l4,l5,l6]
    return SimpleNamespace(**{name:getattr(module,name) for module,functions in zip(modules,LESSON_FUNCTIONS.values()) for name in functions})


def build_toy_repa(config=None):
    return RepaSystem(config or RepaConfig(),practice_ops())
