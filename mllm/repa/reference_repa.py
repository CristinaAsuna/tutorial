"""Independent answers: REPA negative-cosine regularization and linear SiT path."""
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from edu_core.vision import patchify, unpatchify
from edu_core.training import update_ema
from .scaffolding import RepaConfig, RepaSystem


def clean_targets(system, images):
    if images.ndim != 4 or tuple(images.shape[1:]) != (3,16,16):
        raise ValueError('expected clean RGB (B,3,16,16)')
    with torch.no_grad():
        clean = system.codec.encode(images)
        targets = system.teacher(images)
    return clean, targets


def flow_pair(clean, noise, time):
    if clean.shape != noise.shape or time.shape != (clean.shape[0],) or not ((time>=0)&(time<=1)).all():
        raise ValueError('matching clean/noise and time (B,) in [0,1] required')
    t = time[:,None,None,None]
    return (1-t)*clean+t*noise, noise-clean


def conditioned_block(block, tokens, condition):
    shift1,scale1,gate1,shift2,scale2,gate2 = block.modulation(condition).chunk(6,-1)
    modulated = block.norm1(tokens)*(1+scale1[:,None])+shift1[:,None]
    tokens = tokens+gate1[:,None]*block.attention(modulated)
    modulated = block.norm2(tokens)*(1+scale2[:,None])+shift2[:,None]
    return tokens+gate2[:,None]*block.mlp(modulated)


def denoise(model, latent, time, labels, *, project=True):
    tokens = model.patch(patchify(latent,2))+model.positions
    condition = model.time_embed(time[:,None])+model.label_embed(labels)
    projected = None
    for index,block in enumerate(model.blocks,1):
        tokens = model.ops.conditioned_block(block,tokens,condition)
        if project and index == model.config.alignment_depth:
            projected = model.projector(tokens)
    shift,scale = model.final_modulation(condition).chunk(2,-1)
    tokens = model.final_norm(tokens)*(1+scale[:,None])+shift[:,None]
    velocity = unpatchify(model.output(tokens),2,4,grid=(4,4))
    return velocity,projected


def alignment_loss(projected, targets):
    if projected.shape != targets.shape or projected.ndim != 3:
        raise ValueError('paired B,N,D patch targets required')
    return -(F.normalize(projected,dim=-1)*F.normalize(targets.detach(),dim=-1)).sum(-1).mean()


def joint_objective(prediction, velocity, alignment, weight=.5):
    denoising = F.mse_loss(prediction,velocity)
    return {'loss':denoising+weight*alignment,'velocity_loss':denoising,'alignment_loss':alignment}


def ema_update(ema, model, momentum):
    update_ema(ema,model,momentum)


def training_step(system, optimizer, images, labels, *, noise=None, time=None):
    system.train()
    optimizer.zero_grad(set_to_none=True)
    result = system(images,labels,noise=noise,time=time)
    result['loss'].backward()
    optimizer.step()
    system.ops.ema_update(system.ema_denoiser,system.denoiser,system.config.ema_momentum)
    return {key:value.detach() for key,value in result.items()}


@torch.no_grad()
def euler_sample(model, labels, initial_noise, steps):
    if not isinstance(steps,int) or steps<=0:
        raise ValueError('steps must be a positive integer')
    current = initial_noise.clone()
    for index in range(steps):
        time = current.new_full((current.shape[0],),1-index/steps)
        velocity,_ = model(current,time,labels,project=False)
        current = current-velocity/steps
    return current


def decode_latents(codec, latents):
    return codec.decode(latents)


def reference_ops():
    return SimpleNamespace(**{name:globals()[name] for name in (
        'clean_targets','flow_pair','conditioned_block','denoise','alignment_loss',
        'joint_objective','ema_update','training_step','euler_sample','decode_latents')})


def build_toy_repa(config=None, ops=None):
    return RepaSystem(config or RepaConfig(),ops or reference_ops())
