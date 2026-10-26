"""Provided module initialization and batch engineering.
The learning path injects every defining computation through ops. Read this file for
attribute names and layer shapes; no student TODO needs to design a network constructor.
"""
import torch
from torch import nn
from edu_core import MultiHeadAttention


class ConditionalResidual(nn.Module):
    def __init__(self, input_dim, output_dim, ops, cond_dim=48):
        super().__init__()
        self.ops = ops
        def conv(ci, co):
            return nn.Sequential(nn.Conv1d(ci, co, 3, padding=1), nn.GroupNorm(4, co), nn.Mish())
        self.conv1, self.conv2 = conv(input_dim, output_dim), conv(output_dim, output_dim)
        self.condition = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, 2*output_dim))
        self.skip = nn.Conv1d(input_dim, output_dim, 1) if input_dim != output_dim else nn.Identity()

    def forward(self, x, condition):
        return self.ops.residual_forward(self, x, condition)



class ConditionalUNet1D(nn.Module):
    def __init__(self, ops):
        super().__init__()
        self.ops = ops
        self.time_mlp = nn.Sequential(nn.Linear(16, 32), nn.Mish(), nn.Linear(32, 16))
        self.down_block = ConditionalResidual(2, 16, ops)
        self.downsample = nn.Conv1d(16, 32, 4, stride=2, padding=1)
        self.middle = ConditionalResidual(32, 32, ops)
        self.upsample = nn.ConvTranspose1d(32, 16, 4, stride=2, padding=1)
        self.up_block = ConditionalResidual(32, 16, ops)
        self.output = nn.Conv1d(16, 2, 1)

    def forward(self, sample, timesteps, obs):
        return self.ops.unet_forward(self, sample, timesteps, obs)



class DiffusionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(32), nn.LayerNorm(32), nn.LayerNorm(32)
        self.self_attn, self.cross_attn = MultiHeadAttention(32, 4), MultiHeadAttention(32, 4)
        self.mlp = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32))



class TransformerDenoiser(nn.Module):
    def __init__(self, ops):
        super().__init__()
        self.ops = ops
        self.action_in, self.obs_in = nn.Linear(2, 32), nn.Linear(16, 32)
        self.time_mlp = nn.Sequential(nn.Linear(16, 32), nn.Mish(), nn.Linear(32, 32))
        self.action_pos = nn.Parameter(torch.randn(1, 8, 32)*.02)
        self.memory_pos = nn.Parameter(torch.randn(1, 3, 32)*.02)
        self.blocks = nn.ModuleList([DiffusionBlock() for _ in range(2)])
        self.norm, self.output = nn.LayerNorm(32), nn.Linear(32, 2)

    def forward(self, sample, timesteps, obs):
        return self.ops.transformer_forward(self, sample, timesteps, obs)



class DiffusionPolicy(nn.Module):
    def __init__(self, backbone="cnn", ops=None):
        super().__init__()
        if backbone not in ("cnn", "transformer"):
            raise ValueError("unknown backbone")
        if ops is None:
            from reference_diffusion_policy import reference_ops
            ops = reference_ops()
        self.ops = ops
        self.backbone = backbone
        self.obs_horizon, self.pred_horizon, self.action_horizon = 2, 8, 3
        self.action_dim, self.steps = 2, 16
        self.encoder = nn.Sequential(nn.Conv2d(3, 12, 3, padding=1), nn.Mish(),
                                     nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.denoiser = ConditionalUNet1D(self.ops) if backbone == "cnn" else TransformerDenoiser(self.ops)
        betas, alpha_bar = self.ops.cosine_schedule(self.steps)
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("action_low", torch.tensor([-2., -2.]))
        self.register_buffer("action_high", torch.tensor([2., 2.]))

    @property
    def stats(self):
        return {"low": self.action_low, "high": self.action_high}

    def compute_loss(self, batch, noise, timesteps):
        actions = self.ops.normalize(batch["actions"], self.stats)
        noisy = self.ops.add_noise(actions, noise, timesteps, self.alpha_bar)
        obs = self.ops.encode_observations(self.encoder, batch["images"], batch["states"])
        return self.ops.noise_loss(self.denoiser(noisy, timesteps, obs), noise)

    @torch.no_grad()
    def predict_action(self, images, states, initial_noise, step_noises, sampler="ddpm"):
        self.eval()
        normalized = self.ops.sample_actions(self, images, states, initial_noise, step_noises, sampler)
        prediction = self.ops.unnormalize(normalized, self.stats)
        return {"action_pred": prediction,
                "action": self.ops.execution_chunk(prediction, self.obs_horizon, self.action_horizon)}

