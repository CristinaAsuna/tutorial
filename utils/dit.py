"""Diffusion Transformer (DiT) learning scaffold.

All diffusion pipelines in this project use the same contract:

    prediction = model(x_t, timesteps)
    # x_t and prediction: (B, C, H, W)

Therefore this backbone can be used for DDPM (epsilon), VP-SDE (score), or
Flow Matching (velocity).  The output meaning belongs to the *pipeline*, not
to this file.

Implement the TODOs in this order:
1. PatchEmbed and ``unpatchify``; verify their shapes first.
2. TimestepEmbedder.
3. DiTBlock as a normal pre-norm Transformer block.
4. Replace its fixed LayerNorm affine parameters with adaLN-Zero.
5. FinalLayer, zero initialisation, and the DiT forward data flow.
"""
import os
from abc import ABC, abstractmethod
from typing import Optional, List, Type, Tuple, Dict
import math
import uuid
import random

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes._axes import Axes
import torch
import torch.nn as nn
from tqdm import tqdm
from dataclasses import dataclass
# from einops import rearrange
# from einops.layers.torch import Rearrange
import torch
from torch import nn


@dataclass
class DiTConfig:
    """Architecture only; ``image_size`` must be divisible by ``patch_size``."""

    in_channels: int = 3
    image_size: int = 32
    patch_size: int = 2
    hidden_size: int = 384
    depth: int = 8
    num_heads: int = 6
    mlp_ratio: float = 4.0

    @property
    def grid_size(self) -> int:
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.grid_size**2


class PatchEmbed(nn.Module):
    """Convert image patches to tokens.

    input:  (B, C, H, W)
    output: (B, N, D), where N = (H / P) * (W / P)

    TODO: Use ``nn.Conv2d(in_channels, hidden_size, kernel_size=P, stride=P)``
    and flatten its spatial dimensions into the token dimension N.
    """

    def __init__(self, in_channels: int, hidden_size: int, patch_size: int):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size

        # TODO: self.proj = ...
        self.proj=nn.Conv2d(in_channels,hidden_size,kernel_size=patch_size,stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #(b,c,h,w)--->cov2d ---> (b,d,h/p,w/p)
        x=self.proj(x)
        #(b,d,h/p,w/p)-->(b,d,n)
        b,d,h_patch,w_patch=x.shape
        x=x.flatten(2)
        x=x.transpose(1,2)
        #(b,n,d)
        return x



class TimestepEmbedder(nn.Module):
    """Map one scalar time per image to a DiT conditioning vector.

    input:  timesteps (B,) -- DDPM indices or continuous time values
    output: c         (B, D)

    TODO: implement sinusoidal embedding followed by an MLP.  It is fine to
    reuse the *idea* of SinusoidalTimeEmbedding from basic_function.py, but
    keep this module token-backbone-specific and independently testable.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        self.mlp=nn.Sequential(
            nn.Linear(hidden_size,4*hidden_size),
            nn.SiLU(),
            nn.Linear(4*hidden_size,hidden_size)
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim=self.hidden_size//2
        emb=math.log(1000)/(half_dim-1)
        emb=torch.exp(torch.arange(half_dim,device=timesteps.device)*-emb)
        emb=timesteps[:,None].float()*emb[None,:]

        emb=torch.cat([torch.sin(emb),torch.cos(emb)],dim=-1)

        return self.mlp(emb)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply token-wise AdaLN modulation.

    x:     (B, N, D)
    shift: (B, D)
    scale: (B, D)

    TODO: broadcast shift and scale over N, then return
          x * (1 + scale[:, None, :]) + shift[:, None, :].
    """
    return x*(1+scale[:,None,:])+shift[:,None,:]


class DiTBlock(nn.Module):
    """One DiT transformer block with time-conditioned adaLN-Zero.

    x: (B, N, D), c: (B, D)

    The modulation MLP should produce six (B, D) tensors:
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp.

    TODO: use pre-norm self-attention and MLP branches:

        x = x + gate_msa[:, None, :] * attention(modulate(norm1(x), ...))
        x = x + gate_mlp[:, None, :] * mlp(modulate(norm2(x), ...))

    The last linear layer of ``adaLN_modulation`` must be zero-initialized.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        # TODO: self.norm1, self.attn, self.norm2, self.mlp,
        #       self.adaLN_modulation

        self.norm1=nn.LayerNorm(hidden_size,elementwise_affine=False)
        self.norm2=nn.LayerNorm(hidden_size,elementwise_affine=False)

        self.attn=nn.MultiheadAttention(hidden_size,num_heads,batch_first=True)

        mlp_hidden=int(hidden_size*mlp_ratio)

        self.mlp=nn.Sequential(
            nn.Linear(hidden_size,mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden,hidden_size),
        )

        self.adaln=nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size,6*hidden_size),
        )

        nn.init.zeros_(self.adaln[-1].weight)
        nn.init.zeros_(self.adaln[-1].bias)


    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift_msa,scale_msa,gate_msa,shift_mlp,scale_mlp,gate_mlp=self.adaln(condition).chunk(6,dim=1)
        x_norm=self.norm1(x)
        x_mod=modulate(x_norm,shift_msa,scale_msa)

        attn_out,_=self.attn(x_mod,x_mod,x_mod,need_weights=False)
        x=x+gate_msa[:,None,:]*attn_out

        #mlp
        x_norm=self.norm2(x)
        x_mod=modulate(x_norm,shift_mlp,scale_mlp)
        mlp_out=self.mlp(x_mod)
        x=x+gate_mlp[:,None,:]*mlp_out

        return x


class FinalLayer(nn.Module):
    """Map final tokens to pixel values for one patch.

    input:  x (B, N, D), condition (B, D)
    output: (B, N, P*P*C)

    TODO: AdaLN-conditioned LayerNorm followed by a linear projection.  Its
    modulation and output projection should be zero-initialized, as in DiT.
    """

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.out_channels = out_channels

        # 最终LayerNorm（无可学习参数）
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False)

        # adaLN调制：生成shift和scale
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )

        # 线性投影到patch像素值
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)

        # 零初始化调制层和输出层
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        # 生成调制参数
        mod = self.adaLN_modulation(condition)  # (B, 2*D)
        shift, scale = mod.chunk(2, dim=1)

        # 应用调制
        x = self.norm_final(x)
        x = modulate(x, shift, scale)

        # 投影到patch像素
        x = self.linear(x)  # (B, N, P*P*C)
        return x


class DiT(nn.Module):
    """Unconditional image-space Diffusion Transformer.

    forward input/output are deliberately compatible with TimeUNet:
        x_t:       (B, C, H, W)
        timesteps: (B,)
        output:    (B, C, H, W)

    TODO: add a learned or fixed 2-D positional embedding of shape (1, N, D),
    then execute ``patch_embed -> blocks -> final_layer -> unpatchify``.
    """

    def __init__(self, config: DiTConfig):
        super().__init__()
        if config.image_size % config.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if config.hidden_size % config.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.config = config

        # 组件
        self.patch_embed = PatchEmbed(config.in_channels, config.hidden_size, config.patch_size)
        self.time_embed = TimestepEmbedder(config.hidden_size)

        # 可学习的位置嵌入（固定长度，可学习）
        self.pos_embed = nn.Parameter(torch.zeros(1, config.num_patches, config.hidden_size))
        nn.init.normal_(self.pos_embed, std=0.02)

        # Transformer块
        self.blocks = nn.ModuleList([
            DiTBlock(config.hidden_size, config.num_heads, config.mlp_ratio)
            for _ in range(config.depth)
        ])

        # 最终层
        self.final_layer = FinalLayer(config.hidden_size, config.patch_size, config.in_channels)

    def unpatchify(self, patch_values: torch.Tensor) -> torch.Tensor:
        """Convert (B, N, P*P*C) back to (B, C, H, W)."""
        B, N, _ = patch_values.shape
        P = self.config.patch_size
        C = self.config.in_channels
        grid = self.config.grid_size

        # 检查N必须等于grid*grid
        assert N == grid * grid, "Number of patches does not match grid size"

        # 重塑为 (B, grid, grid, P, P, C)
        x = patch_values.view(B, grid, grid, P, P, C)
        # 转置为 (B, C, grid, P, grid, P)
        x = x.permute(0, 5, 1, 3, 2, 4)
        # 重塑为 (B, C, H, W)
        x = x.reshape(B, C, grid * P, grid * P)
        return x

    def forward(self, x_t: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        # 图像 -> tokens
        x = self.patch_embed(x_t)            # (B, N, D)
        x = x + self.pos_embed               # 添加位置嵌入

        # 时间嵌入
        c = self.time_embed(timesteps)       # (B, D)

        # Transformer块
        for block in self.blocks:
            x = block(x, c)

        # 最终层
        x = self.final_layer(x, c)           # (B, N, P*P*C)

        # tokens -> 图像
        x = self.unpatchify(x)               # (B, C, H, W)
        return x


    def get_config(self) -> dict:
        """Lets the existing checkpoint trainer reconstruct this architecture."""
        return {"config": self.config.__dict__.copy()}


if __name__ == "__main__":
    # Keep this as the target test after you finish the TODOs.
    config = DiTConfig(image_size=32, patch_size=2, hidden_size=384, depth=8, num_heads=6)
    model = DiT(config)
    x = torch.randn(2, 3, 32, 32)
    t = torch.randint(0, 1000, (2,))
    y = model(x, t)
    assert y.shape == x.shape
    print("DiT shape check passed:", tuple(y.shape))
