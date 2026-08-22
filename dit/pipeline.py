"""Objective selection for DiT.

DiT is only a backbone: it predicts a tensor shaped like ``x_t``.  The
meaning of that tensor is selected here:

* ``ddpm``: epsilon (noise)
* ``flow_matching``: velocity
* ``vp_sde``: score

Do not duplicate the forward/noising equations in this directory.  These
classes reuse the already-tested implementations in the corresponding method
directories, so one mathematical correction benefits U-Net and DiT equally.
"""

from typing import Literal

from diffusion.pipeline import DDPMPipeline
from flow_matching.pipeline import FlowMatchingPipeline
from score_matching.pipeline import ScoreMatchingPipeline, VPSDE

ObjectiveName = Literal["ddpm", "flow_matching", "vp_sde"]


def build_pipeline(
    objective: ObjectiveName,
    *,
    time_scale: float = 1000.0,
    num_train_steps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    beta_min: float = 0.1,
    beta_max: float = 20.0,
    sde_eps: float = 1e-5,
):
    """Return the target-specific training and sampling pipeline.

    Every returned object supports ``training_loss(model, images)``.  Sampling
    names intentionally remain explicit: DDPM/Flow use ``sample``; VP-SDE uses
    ``sample_euler_maruyama``.
    """
    if objective == "ddpm":
        return DDPMPipeline(
            num_train_steps=num_train_steps,
            beta_start=beta_start,
            beta_end=beta_end,
        )
    if objective == "flow_matching":
        return FlowMatchingPipeline(time_scale=time_scale)
    if objective == "vp_sde":
        return ScoreMatchingPipeline(
            VPSDE(beta_min=beta_min, beta_max=beta_max, eps=sde_eps),
            time_scale=time_scale,
        )
    raise ValueError(f"Unknown objective: {objective!r}")
