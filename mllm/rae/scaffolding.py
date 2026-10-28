"""已提供的模块初始化；关键计算通过 ops 注入，初始化不要求学生重复造轮子。"""
from copy import deepcopy
import torch
from torch import nn
from edu_core.attention import MultiHeadAttention, SelfAttentionBlock
from edu_core.vision import patchify, unpatchify


class FrozenEncoder(nn.Module):
    """随机、冻结的离线fixture，绝不称为预训练语义模型。"""
    def __init__(self, dim=64):
        super().__init__()
        self.proj = nn.Linear(48, dim)
        self.requires_grad_(False)
        self.eval()

    def forward(self, images):
        tokens = self.proj(patchify(images, 4))
        # 模拟CLS+register，使学生必须剥离特殊token。
        return torch.cat((tokens.mean(1, keepdim=True).expand(-1, 2, -1), tokens), 1)


class AdaBlock(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = MultiHeadAttention(dim, heads)
        self.mlp = nn.Sequential(nn.Linear(dim, 2*dim), nn.GELU(), nn.Linear(2*dim, dim))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6*dim))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x, condition):
        if condition.ndim == 2:
            condition = condition[:, None]
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(condition).chunk(6, -1)
        x = x + gate1 * self.attn(self.norm1(x)*(1+scale1)+shift1)
        return x + gate2 * self.mlp(self.norm2(x)*(1+scale2)+shift2)


class WideDDT(nn.Module):
    def __init__(self, ops, latent_dim=64, base_width=32, head_width=96):
        super().__init__()
        self.ops = ops
        self.base_embed = nn.Linear(latent_dim, base_width)
        self.noisy_embed = nn.Linear(latent_dim, head_width)
        self.time_embed = nn.Sequential(nn.Linear(1, base_width), nn.SiLU(), nn.Linear(base_width, base_width))
        self.pos = nn.Parameter(torch.randn(1, 16, base_width)*.02)
        self.base = nn.ModuleList([AdaBlock(base_width) for _ in range(2)])
        self.condition_projector = nn.Linear(base_width, head_width)
        self.head = nn.ModuleList([AdaBlock(head_width) for _ in range(2)])
        self.final_norm = nn.LayerNorm(head_width, elementwise_affine=False)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(head_width, 2*head_width))
        self.out = nn.Linear(head_width, latent_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, noisy, time):
        return self.ops.ddt_forward(self, noisy, time)


class RAE(nn.Module):
    def __init__(self, ops, latent_dim=64):
        super().__init__()
        self.ops = ops
        self.encoder = FrozenEncoder(latent_dim)
        self.decoder_blocks = nn.Sequential(SelfAttentionBlock(latent_dim, 4), SelfAttentionBlock(latent_dim, 4))
        self.decoder_out = nn.Linear(latent_dim, 48)
        self.stage = 1

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        if self.stage == 2:
            super().train(False)
        return self

    def encode(self, images):
        with torch.no_grad():
            return self.ops.normalize_tokens(self.encoder(images), 2)

    def encode_grid(self, images):
        return self.ops.tokens_to_grid(self.encode(images), (4, 4))

    def decode(self, latent):
        return self.ops.decode_tokens(self, latent)

    def forward(self, images, amplitude_normal=None, epsilon=None, tau=.8):
        latent = self.encode(images)
        if self.training:
            latent = self.ops.augment_latent(latent, tau, amplitude_normal, epsilon)
        return self.decode(latent)


class RAECourse(nn.Module):
    def __init__(self, ops):
        super().__init__()
        self.ops = ops
        self.rae = RAE(ops)
        self.generator = WideDDT(ops)
        self.ema = deepcopy(self.generator)
        self.ema.requires_grad_(False).eval()
        self.reference_dim = 4096

    def train(self, mode=True):
        super().train(mode)
        self.rae.train(mode and self.rae.stage == 1)
        self.generator.train(mode and self.rae.stage == 2)
        self.ema.eval()
        return self

    def set_stage(self, stage):
        if stage not in (1, 2):
            raise ValueError('stage must be 1 or 2')
        self.rae.stage = stage
        self.rae.requires_grad_(stage == 1)
        self.rae.encoder.requires_grad_(False)
        self.rae.train(stage == 1)
        self.generator.requires_grad_(stage == 2)
        self.generator.train(stage == 2)
        self.ema.requires_grad_(False).eval()
        for p in self.parameters():
            p.grad = None


def toy_images(batch=2):
    """固定结构图案，无需数据下载。"""
    axis = torch.linspace(-1, 1, 16)
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    image = torch.stack((xx, yy, (xx*yy).sin()))
    return torch.stack([image.roll(i, -1) for i in range(batch)])
