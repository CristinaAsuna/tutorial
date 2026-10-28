"""Provided tiny architecture. Paper-specific computations enter through ops."""
import copy
from dataclasses import dataclass
import torch
from torch import nn
from edu_core.attention import MultiHeadAttention
from edu_core.vision import patchify, unpatchify
from edu_core.training import freeze_and_keep_eval


@dataclass(frozen=True)
class RepaConfig:
    image_size: int = 16
    latent_channels: int = 4
    hidden_dim: int = 32
    teacher_dim: int = 24
    num_classes: int = 4
    depth: int = 2
    alignment_depth: int = 1  # 1-based, after this block
    alignment_weight: float = .5
    ema_momentum: float = .9

    def __post_init__(self):
        if self.image_size != 16 or self.latent_channels != 4:
            raise ValueError('toy contract: images 16px, codec latents 4x8x8')
        if not 1 <= self.alignment_depth <= self.depth or self.alignment_weight < 0:
            raise ValueError('invalid alignment depth/weight')
        if self.hidden_dim <= 0 or self.hidden_dim % 4 or self.teacher_dim <= 0 or self.num_classes <= 0:
            raise ValueError('hidden_dim must divide four heads; dimensions/classes positive')
        if not 0 <= self.ema_momentum <= 1:
            raise ValueError('EMA momentum in [0,1] required')


class ToyCodec(nn.Module):
    """Fixed linear fixture, NOT a pretrained VAE; no posterior/KL claim."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 4, 2, 2)
        self.decoder = nn.ConvTranspose2d(4, 3, 2, 2)

    def encode(self, images):
        return self.encoder(images)

    def decode(self, latents):
        return self.decoder(latents)


class ToyTeacher(nn.Module):
    """Frozen randomly initialized spatial fixture, NOT semantic pretraining."""
    def __init__(self, dim):
        super().__init__()
        self.patch = nn.Conv2d(3, dim, 4, 4)
        self.norm = nn.LayerNorm(dim)

    def forward(self, images):
        return self.norm(self.patch(images).flatten(2).transpose(1, 2))


class ConditionedBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attention = MultiHeadAttention(dim, 4)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(dim, 2*dim), nn.GELU(), nn.Linear(2*dim, dim))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6*dim))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)


class TinySiT(nn.Module):
    def __init__(self, config, ops):
        super().__init__()
        self.config, self.ops = config, ops
        d = config.hidden_dim
        self.patch = nn.Linear(2*2*4, d)
        # Fixed deterministic 2D coordinates, toy alternative to sin-cos positions.
        yy, xx = torch.meshgrid(torch.linspace(-1,1,4), torch.linspace(-1,1,4), indexing='ij')
        coords = torch.stack((yy,xx),-1).reshape(1,16,2)
        self.register_buffer('positions', coords.repeat(1,1,d//2))
        self.time_embed = nn.Sequential(nn.Linear(1,d),nn.SiLU(),nn.Linear(d,d))
        self.label_embed = nn.Embedding(config.num_classes,d)
        self.blocks = nn.ModuleList([ConditionedBlock(d) for _ in range(config.depth)])
        self.projector = nn.Sequential(nn.Linear(d,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,config.teacher_dim))
        self.final_norm = nn.LayerNorm(d,elementwise_affine=False)
        self.final_modulation = nn.Sequential(nn.SiLU(),nn.Linear(d,2*d))
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        self.output = nn.Linear(d,16)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self, latent, time, labels, *, project=True):
        return self.ops.denoise(self,latent,time,labels,project=project)


class RepaSystem(nn.Module):
    def __init__(self, config, ops):
        super().__init__()
        self.config, self.ops = config, ops
        self.codec, self.teacher = ToyCodec(), ToyTeacher(config.teacher_dim)
        self.denoiser = TinySiT(config,ops)
        self.ema_denoiser = copy.deepcopy(self.denoiser)
        for module in (self.codec,self.teacher,self.ema_denoiser):
            freeze_and_keep_eval(module)

    def train(self, mode=True):
        super().train(mode)
        for module in (self.codec,self.teacher,self.ema_denoiser):
            module.eval()
        return self

    def forward(self, images, labels, *, noise=None, time=None):
        clean, targets = self.ops.clean_targets(self,images)
        noise = torch.randn_like(clean) if noise is None else noise
        time = torch.rand(clean.shape[0],device=clean.device) if time is None else time
        noisy, velocity = self.ops.flow_pair(clean,noise,time)
        prediction, projected = self.denoiser(noisy,time,labels)
        alignment = self.ops.alignment_loss(projected,targets)
        return self.ops.joint_objective(prediction,velocity,alignment,self.config.alignment_weight)

    @torch.no_grad()
    def sample(self, labels, initial_noise, steps=8, *, use_ema=True):
        model = self.ema_denoiser if use_ema else self.denoiser
        model.eval()
        latents = self.ops.euler_sample(model,labels,initial_noise,steps)
        return self.ops.decode_latents(self.codec,latents)


def toy_batch(batch_size=4):
    """Structured synthetic RGB/labels, no files or downloads."""
    labels = torch.arange(batch_size) % 4
    yy, xx = torch.meshgrid(torch.linspace(-1,1,16),torch.linspace(-1,1,16),indexing='ij')
    images = torch.stack([torch.stack((xx,yy,(xx*yy)*(1+int(k))*.25)) for k in labels])
    return images, labels
