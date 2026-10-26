"""RSS 2023 Diffusion Policy: CPU conditional action diffusion, two denoisers."""
from __future__ import annotations

import copy
import math
from types import SimpleNamespace
import torch
from edu_core import update_ema


def trajectory_window(observations, actions, now, obs_horizon=2, pred_horizon=8):
    """Single episode only; repeat boundary frames/actions, as in official sampler."""
    start = now - obs_horizon + 1
    oi = torch.arange(start, now + 1).clamp(0, len(observations)-1)
    ai = torch.arange(start, start + pred_horizon).clamp(0, len(actions)-1)
    return observations[oi], actions[ai]


def fit_bounds(data):
    return {"low": data.amin(0), "high": data.amax(0)}


def normalize(data, stats):
    return 2 * (data - stats["low"]) / (stats["high"]-stats["low"]).clamp_min(1e-6) - 1


def unnormalize(data, stats):
    return (data + 1) * .5 * (stats["high"]-stats["low"]) + stats["low"]


def cosine_schedule(steps):
    t = torch.linspace(0, 1, steps+1, dtype=torch.float64)
    cumulative = torch.cos((t+.008)/1.008 * math.pi/2).square()
    cumulative = cumulative / cumulative[0]
    betas = (1-cumulative[1:]/cumulative[:-1]).clamp(max=.999).float()
    return betas, torch.cumprod(1-betas, dim=0)


def add_noise(clean, noise, timesteps, alpha_bar):
    a = alpha_bar[timesteps].reshape(-1, 1, 1)
    return a.sqrt()*clean + (1-a).sqrt()*noise


def encode_observations(encoder, images, states):
    b, to, c, h, w = images.shape
    features = encoder(images.reshape(b*to, c, h, w)).reshape(b, to, -1)
    return torch.cat((features, states), dim=-1)


def timestep_embedding(timesteps, dim):
    frequencies = torch.exp(-math.log(10000)*torch.arange(dim//2, device=timesteps.device)/(dim//2-1))
    angles = timesteps[:, None].float() * frequencies[None]
    return torch.cat((angles.sin(), angles.cos()), dim=-1)


def film_modulate(features, scale_bias):
    scale, bias = scale_bias.chunk(2, dim=-1)
    return scale[:, :, None]*features + bias[:, :, None]


def residual_forward(block, x, condition):
    hidden = block.conv1(x)
    hidden = block.ops.film_modulate(hidden, block.condition(condition))
    return block.conv2(hidden) + block.skip(x)


def unet_down(model, x, cond):
    skip = model.down_block(x, cond)
    return model.downsample(skip), skip


def unet_up(model, x, skip, cond):
    x = model.upsample(x)
    return model.up_block(torch.cat((x, skip), dim=1), cond)


def unet_forward(model, sample, timesteps, obs):
    cond = torch.cat((model.time_mlp(model.ops.timestep_embedding(timesteps, 16)), obs.flatten(1)), dim=-1)
    x, skip = model.ops.unet_down(model, sample.transpose(1, 2), cond)
    x = model.middle(x, cond)
    x = model.ops.unet_up(model, x, skip, cond)
    return model.output(x).transpose(1, 2)


def noise_loss(prediction, noise):
    return (prediction-noise).square().mean()


def training_step(model, ema, batch, optimizer, noise, timesteps, decay=.95):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = model.compute_loss(batch, noise, timesteps)
    loss.backward()
    optimizer.step()
    update_ema(ema, model, decay)
    return loss.detach()


def ddpm_step(sample, prediction, t, betas, alpha_bar, noise):
    a, previous = alpha_bar[t], alpha_bar[t-1] if t else alpha_bar.new_tensor(1.)
    beta, alpha = betas[t], 1-betas[t]
    clean = ((sample - (1-a).sqrt()*prediction)/a.sqrt()).clamp(-1, 1)
    mean = previous.sqrt()*beta/(1-a)*clean + alpha.sqrt()*(1-previous)/(1-a)*sample
    variance = beta*(1-previous)/(1-a)
    return mean + (variance.clamp_min(1e-20).sqrt()*noise if t else 0)


def ddim_step(sample, prediction, t, previous_t, alpha_bar):
    a = alpha_bar[t]
    previous = alpha_bar[previous_t] if previous_t >= 0 else alpha_bar.new_tensor(1.)
    clean = ((sample-(1-a).sqrt()*prediction)/a.sqrt()).clamp(-1, 1)
    return previous.sqrt()*clean + (1-previous).sqrt()*prediction


def sample_actions(model, images, states, initial_noise, step_noises, sampler="ddpm", ddim_steps=4):
    # Cache observation features ONCE, outside the diffusion loop.
    obs = model.ops.encode_observations(model.encoder, images, states)
    x = initial_noise.clone()
    if sampler == "ddpm":
        for t in reversed(range(model.steps)):
            times = torch.full((len(x),), t, device=x.device, dtype=torch.long)
            predicted = model.denoiser(x, times, obs)
            x = model.ops.ddpm_step(x, predicted, t, model.betas, model.alpha_bar, step_noises[t])
    elif sampler == "ddim":
        if not 1 <= ddim_steps <= model.steps:
            raise ValueError("ddim_steps must be in [1, diffusion steps]")
        grid = torch.linspace(model.steps-1, 0, ddim_steps).long().tolist()
        for index, t in enumerate(grid):
            times = torch.full((len(x),), t, device=x.device, dtype=torch.long)
            predicted = model.denoiser(x, times, obs)
            previous = grid[index+1] if index+1 < len(grid) else -1
            x = model.ops.ddim_step(x, predicted, t, previous, model.alpha_bar)
    else:
        raise ValueError("sampler must be ddpm or ddim")
    return x


def execution_chunk(prediction, obs_horizon=2, action_horizon=3):
    return prediction[:, obs_horizon-1:obs_horizon-1+action_horizon]


def receding_rollout(model, observe, execute, rounds, initial_noises, step_noises):
    """Caller adapters own history/physics; each round observes AFTER previous chunk."""
    chunks = []
    for i in range(rounds):
        images, states = observe()
        result = model.predict_action(images, states, initial_noises[i], step_noises[i])
        execute(result["action"])
        chunks.append(result["action"])
    return chunks


def transformer_masks(horizon, obs_horizon, device=None):
    action_allow = torch.ones(horizon, horizon, dtype=torch.bool, device=device).tril()
    # Memory: diffusion time, observation at 0, observation at 1, ...
    memory_allow = torch.arange(horizon, device=device)[:, None] >= (torch.arange(obs_horizon+1, device=device)[None]-1)
    return action_allow, memory_allow


def transformer_tokens(model, sample, timesteps, obs):
    action = model.action_in(sample) + model.action_pos
    time = model.time_mlp(model.ops.timestep_embedding(timesteps, 16))[:, None]
    memory = torch.cat((time, model.obs_in(obs)), dim=1) + model.memory_pos
    return action, memory


def transformer_forward(model, sample, timesteps, obs):
    x, memory = model.ops.transformer_tokens(model, sample, timesteps, obs)
    self_mask, cross_mask = model.ops.transformer_masks(sample.shape[1], obs.shape[1], sample.device)
    for block in model.blocks:
        x = x + block.self_attn(block.norm1(x), attention_mask=self_mask)
        x = x + block.cross_attn(block.norm2(x), memory, attention_mask=cross_mask)
        x = x + block.mlp(block.norm3(x))
    return model.output(model.norm(x))


def reference_ops():
    names = ("trajectory_window", "fit_bounds", "normalize", "unnormalize", "cosine_schedule", "add_noise",
             "encode_observations", "timestep_embedding", "film_modulate", "residual_forward", "unet_down",
             "unet_up", "unet_forward", "noise_loss", "training_step", "ddpm_step", "ddim_step", "sample_actions",
             "execution_chunk", "receding_rollout", "transformer_masks", "transformer_tokens", "transformer_forward")
    return SimpleNamespace(**{name: globals()[name] for name in names})


from scaffolding import ConditionalResidual, ConditionalUNet1D, DiffusionBlock, TransformerDenoiser, DiffusionPolicy

def build_toy_diffusion_policy(backbone: str = "cnn") -> DiffusionPolicy:
    return DiffusionPolicy(backbone)


def make_ema(model):
    return copy.deepcopy(model).requires_grad_(False).eval()
