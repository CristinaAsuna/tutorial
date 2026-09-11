"""Explicit token attention.  This intentionally does not implement RoPE or KV cache."""
from __future__ import annotations

import math
import torch
from torch import nn

from .masks import causal_allow_mask, combine_allow_masks, key_padding_allow_mask


class MultiHeadAttention(nn.Module):
    """Self/cross attention with a single unambiguous mask convention.

    ``key_padding_mask`` is `(B, K)` and ``attention_mask`` is `(Q, K)` or
    `(B, Q, K)`; True means that a key may be attended to.
    """
    def __init__(self, dim: int, num_heads: int, *, kv_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        if dim <= 0 or num_heads <= 0 or dim % num_heads:
            raise ValueError("dim must be positive and divisible by num_heads")
        self.dim, self.num_heads, self.head_dim = dim, num_heads, dim // num_heads
        kv_dim = dim if kv_dim is None else kv_dim
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj, self.v_proj = nn.Linear(kv_dim, dim), nn.Linear(kv_dim, dim)
        self.out_proj, self.dropout = nn.Linear(dim, dim), nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, context: torch.Tensor | None = None, *,
                key_padding_mask: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None, causal: bool = False) -> torch.Tensor:
        if query.ndim != 3:
            raise ValueError("query must have shape (B, Q, D)")
        context = query if context is None else context
        if context.ndim != 3 or context.shape[0] != query.shape[0]:
            raise ValueError("context must have shape (B, K, D_context) with matching batch")
        b, q_len, _ = query.shape
        k_len = context.shape[1]
        q = self.q_proj(query).view(b, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(b, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(b, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_allow_mask(key_padding_mask)
            if key_padding_mask.shape != (b, k_len):
                raise ValueError("key_padding_mask must have shape (B, K)")
            key_padding_mask = key_padding_mask[:, None, None, :]
        if attention_mask is not None:
            if attention_mask.ndim not in (2, 3) or attention_mask.shape[-2:] != (q_len, k_len):
                raise ValueError("attention_mask must have shape (Q,K) or (B,Q,K)")
            attention_mask = attention_mask.bool().unsqueeze(1) if attention_mask.ndim == 3 else attention_mask.bool()[None, None]
        allowed = combine_allow_masks(key_padding_mask, attention_mask,
                                      causal_allow_mask(q_len, k_len, device=query.device)[None, None] if causal else None)
        if allowed is not None:
            scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        weights = self.dropout(scores.softmax(dim=-1))
        return self.out_proj((weights @ v).transpose(1, 2).reshape(b, q_len, self.dim))


class TransformerBlock(nn.Module):
    """Pre-LN block; optional cross-attention is activated only with context."""
    def __init__(self, dim: int, num_heads: int, *, mlp_ratio: float = 4.0, kv_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1, self.self_attn = nn.LayerNorm(dim), MultiHeadAttention(dim, num_heads, dropout=dropout)
        self.norm2, self.cross_attn = nn.LayerNorm(dim), MultiHeadAttention(dim, num_heads, kv_dim=kv_dim, dropout=dropout)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor, *, key_padding_mask=None, causal=False, context=None, context_padding_mask=None, attention_mask=None):
        x = x + self.self_attn(self.norm1(x), key_padding_mask=key_padding_mask, attention_mask=attention_mask, causal=causal)
        if context is not None:
            x = x + self.cross_attn(self.norm2(x), context, key_padding_mask=context_padding_mask)
        return x + self.mlp(self.norm3(x))
