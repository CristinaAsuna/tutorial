"""Inspectable projected attention/RoPE primitives; no robotics-specific routing."""
import torch
from torch import nn


class RMSNorm(nn.Module):
    """Parameter-compatible RMS normalization using primitives available in torch 2.1."""
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x*torch.rsqrt(x.square().mean(-1, keepdim=True)+torch.finfo(x.dtype).eps)*self.weight


def rotary(x, positions, theta=10000.):
    """Llama-style rotate_half, x=(B,H,L,D), positions=(B,L)."""
    inverse = theta ** (-torch.arange(0, x.shape[-1], 2, device=x.device).float()/x.shape[-1])
    phase = positions.float()[..., None]*inverse
    angles = torch.cat((phase, phase), -1)[:, None]
    left, right = x.chunk(2, -1)
    rotated = torch.cat((-right, left), -1)
    return x*angles.cos().to(x.dtype)+rotated*angles.sin().to(x.dtype)


def projected_attention(q, k, v, allow):
    """GQA with (B,Q,K) allow mask; fully masked rows contribute zero."""
    repeats = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(repeats, 1), v.repeat_interleave(repeats, 1)
    scores = q @ k.transpose(-2, -1) / q.shape[-1]**.5
    permitted = allow[:, None].bool()
    weights = scores.masked_fill(~permitted, torch.finfo(scores.dtype).min).softmax(-1)
    weights = weights * permitted.any(-1, keepdim=True)
    return (weights @ v).transpose(1, 2).flatten(2)


class DecoderLayer(nn.Module):
    """Provided parameter scaffold; hidden width may differ from total attention width."""
    def __init__(self, spec, hidden_size=None):
        super().__init__()
        self.spec = spec
        dim = spec.hidden_size if hidden_size is None else hidden_size
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.q_proj = nn.Linear(dim, spec.num_heads*spec.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, spec.num_kv_heads*spec.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, spec.num_kv_heads*spec.head_dim, bias=False)
        self.o_proj = nn.Linear(spec.num_heads*spec.head_dim, dim, bias=False)
        self.mlp = nn.Sequential(nn.Linear(dim, 2*dim), nn.SiLU(), nn.Linear(2*dim, dim))

    def project(self, x, positions):
        normed = self.norm1(x)
        def heads(proj, count):
            return proj(normed).reshape(x.shape[0], x.shape[1], count, self.spec.head_dim).transpose(1, 2)
        q = rotary(heads(self.q_proj, self.spec.num_heads), positions, self.spec.rope_theta)
        k = rotary(heads(self.k_proj, self.spec.num_kv_heads), positions, self.spec.rope_theta)
        return q, k, heads(self.v_proj, self.spec.num_kv_heads)

    def finish(self, x, attended):
        x = x+self.o_proj(attended)
        return x+self.mlp(self.norm2(x))
