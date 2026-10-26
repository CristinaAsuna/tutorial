"""Explicit lesson composition. Reference classes supply initialization/outer routing only."""
from types import SimpleNamespace
from scaffolding import DiffusionPolicy
import lesson1_trajectory_windows
import lesson2_forward_diffusion
import lesson3_observation_condition
import lesson4_conditional_unet
import lesson5_noise_training
import lesson6_reverse_sampling
import lesson7_receding_control
import lesson8_transformer_extension


def practice_ops():
    return SimpleNamespace(
        trajectory_window=lesson1_trajectory_windows.trajectory_window,
        fit_bounds=lesson1_trajectory_windows.fit_bounds,
        normalize=lesson1_trajectory_windows.normalize,
        unnormalize=lesson1_trajectory_windows.unnormalize,
        cosine_schedule=lesson2_forward_diffusion.cosine_schedule,
        add_noise=lesson2_forward_diffusion.add_noise,
        encode_observations=lesson3_observation_condition.encode_observations,
        timestep_embedding=lesson4_conditional_unet.timestep_embedding,
        film_modulate=lesson4_conditional_unet.film_modulate,
        residual_forward=lesson4_conditional_unet.residual_forward,
        unet_down=lesson4_conditional_unet.unet_down,
        unet_up=lesson4_conditional_unet.unet_up,
        unet_forward=lesson4_conditional_unet.unet_forward,
        noise_loss=lesson5_noise_training.noise_loss,
        training_step=lesson5_noise_training.training_step,
        ddpm_step=lesson6_reverse_sampling.ddpm_step,
        ddim_step=lesson6_reverse_sampling.ddim_step,
        sample_actions=lesson6_reverse_sampling.sample_actions,
        execution_chunk=lesson7_receding_control.execution_chunk,
        receding_rollout=lesson7_receding_control.receding_rollout,
        transformer_masks=lesson8_transformer_extension.transformer_masks,
        transformer_tokens=lesson8_transformer_extension.transformer_tokens,
        transformer_forward=lesson8_transformer_extension.transformer_forward,
    )


def build_toy_diffusion_policy(backbone: str = "cnn") -> DiffusionPolicy:
    return DiffusionPolicy(backbone, ops=practice_ops())
