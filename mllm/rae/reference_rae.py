"""RAE paper-v1 参考机制；不是官方checkpoint兼容实现。"""
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from edu_core.vision import unpatchify
from edu_core.training import update_ema
from .scaffolding import RAECourse


def normalize_tokens(tokens, special_tokens=2):
    if tokens.ndim != 3 or not 0 <= special_tokens < tokens.shape[1]:
        raise ValueError('tokens must be B,N,D with valid special-token count')
    patches = tokens[:, special_tokens:]
    return F.layer_norm(patches, (patches.shape[-1],), weight=None, bias=None)


def tokens_to_grid(tokens, grid=(4, 4)):
    b, n, d = tokens.shape
    if n != grid[0]*grid[1]:
        raise ValueError('grid must match token count')
    return tokens.transpose(1, 2).reshape(b, d, *grid)


def grid_to_tokens(grid):
    return grid.flatten(2).transpose(1, 2)


def decode_tokens(model, latent):
    return unpatchify(model.decoder_out(model.decoder_blocks(latent)), 4, 3, grid=(4, 4))


def augment_latent(latent, tau=.8, amplitude_normal=None, epsilon=None):
    if tau < 0:
        raise ValueError('tau must be nonnegative')
    if amplitude_normal is None:
        amplitude_normal = torch.randn(latent.shape[0], device=latent.device, dtype=latent.dtype)
    if epsilon is None:
        epsilon = torch.randn_like(latent)
    if amplitude_normal.shape != (latent.shape[0],) or epsilon.shape != latent.shape:
        raise ValueError('amplitude_normal B and epsilon B,N,D required')
    sigma = tau*amplitude_normal.abs()[:, None, None]
    return latent + sigma*epsilon


def reconstruction_loss(prediction, images):
    return (prediction-images).abs().mean()


def stage1_step(course, images, optimizer):
    course.set_stage(1)
    optimizer.zero_grad(set_to_none=True)
    prediction = course.rae(images)
    loss = course.ops.reconstruction_loss(prediction, images)
    loss.backward()
    optimizer.step()
    return loss.detach()


def shift_time(time, effective_dim, reference_dim=4096):
    if effective_dim <= 0 or reference_dim <= 0 or ((time < 0) | (time > 1)).any():
        raise ValueError('positive dimensions and time in [0,1] required')
    alpha = (effective_dim/reference_dim)**.5
    return alpha*time/(1+(alpha-1)*time)


def flow_pair(latent, time, noise):
    if latent.shape != noise.shape or time.shape != (latent.shape[0],):
        raise ValueError('latent/noise B,N,D and time B required')
    t = time[:, None, None]
    return (1-t)*latent+t*noise, noise-latent


def velocity_loss(prediction, target):
    return (prediction-target.detach()).square().mean()


def ddt_forward(model, noisy, time):
    t = model.time_embed(time[:, None])
    s = model.base_embed(noisy)+model.pos
    for block in model.base:
        s = block(s, F.silu(t))
    condition = model.condition_projector(F.silu(t[:, None]+s))
    x = model.noisy_embed(noisy)
    for block in model.head:
        x = block(x, condition)
    shift, scale = model.final_modulation(condition).chunk(2, -1)
    return model.out(model.final_norm(x)*(1+scale)+shift)


def stage2_step(course, images, optimizer, time=None, noise=None, momentum=.9):
    course.set_stage(2)
    optimizer.zero_grad(set_to_none=True)
    latent = course.ops.grid_to_tokens(course.rae.encode_grid(images))
    if time is None:
        time = torch.rand(images.shape[0], device=images.device)
    time = course.ops.shift_time(time, latent.shape[1]*latent.shape[2], course.reference_dim)
    if noise is None:
        noise = torch.randn_like(latent)
    noisy, target = course.ops.flow_pair(latent, time, noise)
    prediction = course.generator(noisy, time)
    loss = course.ops.velocity_loss(prediction, target)
    loss.backward()
    optimizer.step()
    update_ema(course.ema, course.generator, momentum)
    return loss.detach()


@torch.no_grad()
def euler_sample(model, initial_noise, steps=8):
    if steps <= 0:
        raise ValueError('steps must be positive')
    x = initial_noise.clone()
    grid = model.ops.shift_time(torch.linspace(1, 0, steps+1, device=x.device, dtype=x.dtype),
                                x.shape[1]*x.shape[2], 4096)
    for step in range(steps):
        t = grid[step].expand(x.shape[0])
        x = x + (grid[step+1]-grid[step])*model(x, t)
    return x


@torch.no_grad()
def sample_images(course, initial_noise, steps=8):
    course.set_stage(2)
    latent = course.ops.euler_sample(course.ema, initial_noise, steps)
    return course.rae.decode(latent)


def reference_ops():
    names = ('tokens_to_grid', 'grid_to_tokens', 'normalize_tokens', 'decode_tokens', 'augment_latent', 'reconstruction_loss', 'stage1_step',
             'shift_time', 'flow_pair', 'velocity_loss', 'ddt_forward', 'stage2_step', 'euler_sample', 'sample_images')
    return SimpleNamespace(**{name: globals()[name] for name in names})


def build_toy_rae():
    model = RAECourse(reference_ops())
    model.set_stage(1)
    return model
