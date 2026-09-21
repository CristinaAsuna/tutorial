"""Flamingo core mechanisms on CPU-sized, inspectable toy tensors.

This is not checkpoint-compatible Flamingo.  It keeps the architectural
distinction that matters for study: images become a fixed visual memory through
a Perceiver Resampler, then frozen language layers read that memory through
zero-gated cross-attention.  Visual memory is *not* inserted into the text
embedding sequence as it is in LLaVA.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from edu_core.attention import GatedCrossAttentionBlock, MultiHeadAttention, TransformerBlock
from edu_core.training import freeze_and_keep_eval, set_requires_grad

IMAGE_TOKEN_INDEX = -200
IGNORE_INDEX = -100


class MockVisionEncoder(nn.Module):
    """Tiny frozen ViT-like image encoder returning patch features only."""
    def __init__(self, image_size: int = 8, patch_size: int = 4, vision_dim: int = 24):
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size, self.patch_size = image_size, patch_size
        self.patch_embed = nn.Conv2d(3, vision_dim, patch_size, patch_size)
        self.num_patches = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, vision_dim))
        self.norm = nn.LayerNorm(vision_dim)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must be (B, 3, H, W)")
        if tuple(images.shape[-2:]) != (self.image_size, self.image_size):
            raise ValueError(f"expected {self.image_size}x{self.image_size} images")
        return self.norm(self.patch_embed(images).flatten(2).transpose(1, 2) + self.pos_embed)


class PerceiverResampler(nn.Module):
    """Compress each image's variable patch set to a fixed number of latents."""
    def __init__(self, vision_dim: int, dim: int, *, num_latents: int = 4, num_heads: int = 4, depth: int = 2):
        super().__init__()
        if num_latents <= 0:
            raise ValueError("num_latents must be positive")
        self.vision_proj = nn.Linear(vision_dim, dim)
        self.latents = nn.Parameter(torch.randn(1, num_latents, dim) * 0.02)
        self.cross_norm = nn.LayerNorm(dim)
        self.cross_attn = MultiHeadAttention(dim, num_heads)
        self.blocks = nn.ModuleList([TransformerBlock(dim, num_heads, mlp_ratio=2.0) for _ in range(depth)])

    def forward(self, visual_features: Tensor) -> Tensor:
        if visual_features.ndim != 4:
            raise ValueError("visual_features must be (B, num_images, num_patches, vision_dim)")
        b, images, patches, _ = visual_features.shape
        if images <= 0 or patches <= 0:
            raise ValueError("each sample needs at least one image and one patch")
        memory = self.vision_proj(visual_features).reshape(b * images, patches, -1)
        latents = self.latents.expand(b * images, -1, -1)
        # Learned latents read the complete visual feature set, then refine
        # themselves; the resulting count is independent of patch count.
        latents = latents + self.cross_attn(self.cross_norm(latents), memory)
        for block in self.blocks:
            latents = block(latents)
        return latents.reshape(b, images, latents.shape[1], latents.shape[2])


class FlamingoDecoder(nn.Module):
    """Frozen causal LM layers with trainable Flamingo connector insertion."""
    def __init__(self, vocab_size: int, dim: int, *, max_positions: int = 64, num_heads: int = 4,
                 depth: int = 4, cross_attention_every: int = 2):
        super().__init__()
        if cross_attention_every <= 0:
            raise ValueError("cross_attention_every must be positive")
        self.dim, self.max_positions = dim, max_positions
        self.embed_tokens = nn.Embedding(vocab_size, dim)
        self.position_embed = nn.Embedding(max_positions, dim)
        self.blocks = nn.ModuleList([TransformerBlock(dim, num_heads, mlp_ratio=2.0) for _ in range(depth)])
        self.gated_cross_attn = nn.ModuleDict({
            str(index): GatedCrossAttentionBlock(dim, num_heads, mlp_ratio=2.0)
            for index in range(depth) if (index + 1) % cross_attention_every == 0
        })
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def freeze_backbone(self) -> None:
        for module in (self.embed_tokens, self.position_embed, self.blocks, self.norm, self.lm_head):
            freeze_and_keep_eval(module)

    def keep_backbone_eval(self) -> None:
        for module in (self.embed_tokens, self.position_embed, self.blocks, self.norm, self.lm_head):
            module.eval()

    def forward(self, input_ids: Tensor, visual_memory: Tensor, media_attention_mask: Tensor, attention_mask: Tensor) -> Tensor:
        if input_ids.ndim != 2 or visual_memory.ndim != 4:
            raise ValueError("input_ids must be (B,L) and visual_memory must be (B,M,R,D)")
        b, length = input_ids.shape
        if length > self.max_positions or visual_memory.shape[0] != b:
            raise ValueError("invalid batch size or sequence length")
        if media_attention_mask.shape != (b, length, visual_memory.shape[1] * visual_memory.shape[2]):
            raise ValueError("media_attention_mask must align text positions with flattened visual memory")
        if attention_mask.shape != (b, length):
            raise ValueError("attention_mask must have shape (B,L)")
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0)
        x = self.embed_tokens(input_ids) + self.position_embed(positions)
        memory = visual_memory.flatten(1, 2)
        for index, block in enumerate(self.blocks):
            x = block(x, key_padding_mask=attention_mask, causal=True)
            key = str(index)
            if key in self.gated_cross_attn:
                x = self.gated_cross_attn[key](x, memory, attention_mask=media_attention_mask)
        return self.lm_head(self.norm(x))


class FlamingoForConditionalGeneration(nn.Module):
    """Frozen vision/LM backbones plus trainable Resampler and GCA connectors."""
    def __init__(self, vision_encoder: MockVisionEncoder, resampler: PerceiverResampler,
                 decoder: FlamingoDecoder, *, image_token_id: int = 3, pad_token_id: int = 0):
        super().__init__()
        if resampler.vision_proj.in_features != vision_encoder.pos_embed.shape[-1] or resampler.vision_proj.out_features != decoder.dim:
            raise ValueError("vision, resampler, and decoder dimensions must agree")
        self.vision_encoder, self.resampler, self.decoder = vision_encoder, resampler, decoder
        self.image_token_id, self.pad_token_id = image_token_id, pad_token_id
        self.set_training_stage("connectors")

    def set_training_stage(self, stage: str) -> None:
        if stage != "connectors":
            raise ValueError("this paper-faithful toy supports only the frozen-backbone 'connectors' stage")
        freeze_and_keep_eval(self.vision_encoder)
        self.decoder.freeze_backbone()
        set_requires_grad(self.resampler, True)
        set_requires_grad(self.decoder.gated_cross_attn, True)

    def train(self, mode: bool = True):
        super().train(mode)
        self.vision_encoder.eval()
        self.decoder.keep_backbone_eval()
        return self

    def encode_images(self, pixel_values: Tensor) -> Tensor:
        if pixel_values.ndim != 5:
            raise ValueError("pixel_values must be (B, num_images, C, H, W)")
        b, images = pixel_values.shape[:2]
        with torch.no_grad():
            features = self.vision_encoder(pixel_values.flatten(0, 1))
        return self.resampler(features.reshape(b, images, features.shape[1], features.shape[2]))

    def build_media_attention_mask(self, input_ids: Tensor, num_images: int, attention_mask: Tensor) -> Tensor:
        """Return ``(B,L,M*R)`` visibility: a token reads images to its left only."""
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must both be (B,L)")
        if num_images <= 0:
            raise ValueError("num_images must be positive")
        valid = attention_mask.bool()
        sentinels = (input_ids == IMAGE_TOKEN_INDEX) & valid
        if not torch.all(sentinels.sum(dim=1) == num_images):
            raise ValueError("each sample must contain exactly one valid image sentinel per image")
        if ((input_ids < 0) & (input_ids != IMAGE_TOKEN_INDEX) & valid).any():
            raise ValueError("only IMAGE_TOKEN_INDEX may be negative in valid input_ids")
        image_number = sentinels.long().cumsum(dim=1)
        per_image = torch.arange(num_images, device=input_ids.device).view(1, 1, num_images) < image_number.unsqueeze(-1)
        return per_image.repeat_interleave(self.resampler.latents.shape[1], dim=-1) & valid.unsqueeze(-1)

    def _lm_ids(self, input_ids: Tensor) -> Tensor:
        ids = input_ids.masked_fill(input_ids == IMAGE_TOKEN_INDEX, self.image_token_id)
        if (ids < 0).any() or (ids >= self.decoder.embed_tokens.num_embeddings).any():
            raise ValueError("non-sentinel token id is outside the vocabulary")
        return ids

    def forward(self, input_ids: Tensor, pixel_values: Tensor, *, attention_mask: Optional[Tensor] = None,
                labels: Optional[Tensor] = None) -> dict[str, Tensor | None]:
        if input_ids.ndim != 2 or pixel_values.ndim != 5 or input_ids.shape[0] != pixel_values.shape[0]:
            raise ValueError("input_ids (B,L) and pixel_values (B,M,C,H,W) need matching batches")
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError("labels must match input_ids")
        media_mask = self.build_media_attention_mask(input_ids, pixel_values.shape[1], attention_mask)
        memory = self.encode_images(pixel_values)
        logits = self.decoder(self._lm_ids(input_ids), memory, media_mask, attention_mask)
        loss = None
        if labels is not None:
            labels = labels.masked_fill(input_ids == IMAGE_TOKEN_INDEX, IGNORE_INDEX).masked_fill(~attention_mask, IGNORE_INDEX)
            if not (labels[:, 1:] != IGNORE_INDEX).any():
                raise ValueError("at least one non-padding target token is required")
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=IGNORE_INDEX)
        return {"loss": loss, "logits": logits, "visual_memory": memory, "media_attention_mask": media_mask, "labels": labels}

    @torch.no_grad()
    def generate(self, input_ids: Tensor, pixel_values: Tensor, *, attention_mask: Optional[Tensor] = None,
                 max_new_tokens: int = 4) -> Tensor:
        self.eval()
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
        rows = []
        for row in range(input_ids.shape[0]):
            ids = input_ids[row:row + 1, attention_mask[row]].clone()
            images = pixel_values[row:row + 1]
            for _ in range(max_new_tokens):
                out = self(ids, images, attention_mask=torch.ones_like(ids, dtype=torch.bool))
                next_id = out["logits"][:, -1].argmax(dim=-1, keepdim=True)
                ids = torch.cat((ids, next_id), dim=1)
            rows.append(ids[:, -max_new_tokens:].squeeze(0))
        return torch.stack(rows)


def build_toy_flamingo(vocab_size: int = 64) -> FlamingoForConditionalGeneration:
    vision = MockVisionEncoder(image_size=8, patch_size=4, vision_dim=24)
    resampler = PerceiverResampler(24, 32, num_latents=3, num_heads=4, depth=1)
    decoder = FlamingoDecoder(vocab_size, 32, max_positions=64, num_heads=4, depth=4, cross_attention_every=2)
    return FlamingoForConditionalGeneration(vision, resampler, decoder, image_token_id=3, pad_token_id=0)
