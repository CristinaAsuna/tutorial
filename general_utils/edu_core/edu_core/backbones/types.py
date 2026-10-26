"""Small capability-oriented contracts. True always means valid/allowed."""
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
import torch
from torch import Tensor


@dataclass(frozen=True)
class DecoderSpec:
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    rope_theta: float = 10000.

    def __post_init__(self):
        if min(self.hidden_size, self.num_layers, self.num_heads, self.num_kv_heads, self.head_dim, self.vocab_size) <= 0:
            raise ValueError("decoder dimensions must be positive")
        if self.num_heads % self.num_kv_heads or self.head_dim % 2:
            raise ValueError("GQA heads must divide query heads; rotary head dimension must be even")


@dataclass
class VisionFeatures:
    tokens: Tensor  # (B,N,D); raw patch or language-projected, explicitly selected
    valid_mask: Tensor  # (B,N)
    grid: tuple[int, int] | None
    cls_index: int | None = None


@dataclass
class LanguageOutput:
    hidden_states: Tensor
    logits: Tensor | None
    key_values: tuple[tuple[Tensor, Tensor], ...] = ()


@dataclass(frozen=True)
class PrefixEncoding:
    """Fresh per observation. Keys already have RoPE; NEVER append suffix keys here."""
    key_values: tuple[tuple[Tensor, Tensor], ...]  # layer -> (B,Hkv,P,Dh)
    valid_mask: Tensor
    positions: Tensor


@runtime_checkable
class Backbone(Protocol):
    spec: DecoderSpec
    vision_dim: int
    capabilities: frozenset[str]

    def embed_text(self, ids: Tensor) -> Tensor: ...
    def encode_vision(self, images: Tensor, *, projected: bool = False) -> VisionFeatures: ...
    def language(self, embeddings: Tensor, valid_mask: Tensor, *, allow_mask=None,
                 positions=None, use_cache=False, output_logits=True) -> LanguageOutput: ...


def require_capabilities(backbone, *names):
    missing = set(names) - backbone.capabilities
    if missing:
        raise ValueError(f"backbone lacks required capabilities: {sorted(missing)}")


def valid_positions(valid_mask):
    return (valid_mask.long().cumsum(-1)-1).clamp_min(0)


def prefix_encoding(backbone, embeddings, valid_mask):
    require_capabilities(backbone, "block_mask", "layer_kv")
    allow = valid_mask[:, :, None] & valid_mask[:, None, :]
    positions = valid_positions(valid_mask)
    result = backbone.language(embeddings, valid_mask, allow_mask=allow,
                               positions=positions, use_cache=True, output_logits=False)
    return PrefixEncoding(result.key_values, valid_mask, positions)
