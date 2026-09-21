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
        if allowed is not None:
            # ``softmax([min, ..., min])`` is uniform, not empty.  Queries
            # whose mask exposes no key must instead contribute zero.
            weights = weights * allowed.any(dim=-1, keepdim=True).to(weights.dtype)
        output = self.out_proj((weights @ v).transpose(1, 2).reshape(b, q_len, self.dim))
        if allowed is not None:
            # Also remove the output-projection bias on an empty attention row.
            output = output * allowed.any(dim=-1).any(dim=1).unsqueeze(-1).to(output.dtype)
        return output


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


class GatedCrossAttentionBlock(nn.Module):
    """Cross-attend to an external token memory through zero-initialized gates.

    The block is intentionally model-neutral: a caller supplies the memory and
    its allowed-key mask.  ``attention_mask`` follows this package's convention
    of ``True == allowed`` and may be ``(B, Q, K)`` when every query has a
    different visible part of the memory.
    """
    def __init__(self, dim: int, num_heads: int, *, kv_dim: int | None = None, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.cross_attn = MultiHeadAttention(dim, num_heads, kv_dim=kv_dim, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim))
        # Flamingo-style zero gates preserve the frozen backbone at step zero.
        self.attn_gate = nn.Parameter(torch.zeros(()))
        self.ff_gate = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor, context: torch.Tensor, *, context_padding_mask=None, attention_mask=None) -> torch.Tensor:
        attended = self.cross_attn(self.norm1(x), context, key_padding_mask=context_padding_mask,
                                   attention_mask=attention_mask)
        x = x + self.attn_gate.tanh() * attended
        return x + self.ff_gate.tanh() * self.mlp(self.norm2(x))
