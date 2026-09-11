"""LLaVA-1.5 core mechanics, written for CPU-sized teaching examples.

This is deliberately a mock CLIP + decoder-only LLM: it demonstrates data
flow, not a checkpoint-compatible reproduction of the original project.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from edu_core.training import freeze_and_keep_eval, set_requires_grad

IMAGE_TOKEN_INDEX = -200
IGNORE_INDEX = -100


class MockVisionEncoder(nn.Module):
    """Tiny ViT-like encoder returning ``[CLS] + patch`` features."""

    def __init__(self, image_size=8, patch_size=4, in_chans=3, hidden_size=24):
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size, self.patch_size = image_size, patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_chans, hidden_size, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, hidden_size))
        layer = nn.TransformerEncoderLayer(hidden_size, 4, hidden_size * 2,
                                           batch_first=True, dropout=0.0, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 4:
            raise ValueError("pixel_values must be (B, C, H, W)")
        b, _, h, w = pixel_values.shape
        if h != self.image_size or w != self.image_size:
            raise ValueError(f"expected {self.image_size}x{self.image_size} images")
        patches = self.patch_embed(pixel_values).flatten(2).transpose(1, 2)
        x = torch.cat((self.cls_token.expand(b, -1, -1), patches), dim=1)
        return self.encoder(x + self.pos_embed)


class LlavaProjector(nn.Module):
    """The LLaVA-1.5 two-layer visual-to-language MLP."""

    def __init__(self, vision_dim: int, llm_dim: int):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(vision_dim, llm_dim), nn.GELU(), nn.Linear(llm_dim, llm_dim))

    def forward(self, patch_features: torch.Tensor) -> torch.Tensor:
        return self.layers(patch_features)


class MockDecoderLM(nn.Module):
    """A small causal decoder that accepts either ids or pre-built embeddings."""

    def __init__(self, vocab_size=64, hidden_size=32, max_positions=128, num_heads=4):
        super().__init__()
        self.hidden_size, self.max_positions = hidden_size, max_positions
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.position_embed = nn.Embedding(max_positions, hidden_size)
        layer = nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_size * 2,
                                           batch_first=True, dropout=0.0, activation="gelu")
        self.decoder = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: Optional[torch.Tensor] = None, *,
                inputs_embeds: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids or inputs_embeds")
        x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if x.ndim != 3:
            raise ValueError("decoder inputs must be (B, L, D)")
        b, length, d = x.shape
        if d != self.hidden_size or length > self.max_positions:
            raise ValueError("invalid hidden size or sequence exceeds max_positions")
        if attention_mask is None:
            attention_mask = torch.ones(b, length, dtype=torch.bool, device=x.device)
        if attention_mask.shape != (b, length):
            raise ValueError("attention_mask must have shape (B, L)")
        attention_mask = attention_mask.bool()
        pos = torch.arange(length, device=x.device).unsqueeze(0)
        x = x + self.position_embed(pos)
        # True means blocked for TransformerEncoder's attention mask.
        causal = torch.triu(torch.ones(length, length, device=x.device, dtype=torch.bool), diagonal=1)
        x = self.decoder(x, mask=causal, src_key_padding_mask=~attention_mask)
        return self.lm_head(self.norm(x))


class LlavaForConditionalGeneration(nn.Module):
    def __init__(self, vision_encoder: MockVisionEncoder, llm: MockDecoderLM, projector: LlavaProjector,
                 pad_token_id: int = 0):
        super().__init__()
        if projector.layers[0].in_features != vision_encoder.cls_token.shape[-1]:
            raise ValueError("projector vision dimension does not match encoder")
        if projector.layers[-1].out_features != llm.hidden_size:
            raise ValueError("projector LLM dimension does not match decoder")
        self.vision_encoder, self.llm, self.projector = vision_encoder, llm, projector
        self.pad_token_id = pad_token_id
        self.freeze_vision()

    def freeze_vision(self) -> None:
        freeze_and_keep_eval(self.vision_encoder)

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen CLIP must remain deterministic/eval even during LLaVA training.
        self.vision_encoder.eval()
        return self

    def set_training_stage(self, stage: str) -> None:
        if stage not in {"pretrain_projector", "instruction_tuning"}:
            raise ValueError("stage must be pretrain_projector or instruction_tuning")
        self.freeze_vision()
        set_requires_grad(self.projector, True)
        set_requires_grad(self.llm, stage == "instruction_tuning")

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # No gradient is required through the frozen vision tower. Discard CLS.
        with torch.no_grad():
            features = self.vision_encoder(pixel_values)
        return self.projector(features[:, 1:, :])

    def pack_multimodal_inputs(self, input_ids: torch.Tensor, image_features: torch.Tensor,
                               attention_mask: Optional[torch.Tensor] = None,
                               labels: Optional[torch.Tensor] = None):
        """Replace one -200 sentinel per row with its contiguous patch embeddings."""
        if input_ids.ndim != 2 or image_features.ndim != 3:
            raise ValueError("input_ids must be (B,L); image_features must be (B,N,D)")
        b, text_len = input_ids.shape
        if image_features.shape[0] != b:
            raise ValueError("image batch size must match input batch size")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        if not torch.all((attention_mask == 0) | (attention_mask == 1)):
            raise ValueError("attention_mask values must be 0 or 1")
        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError("labels must match input_ids before image expansion")
        if (input_ids < 0).logical_and(input_ids != IMAGE_TOKEN_INDEX).any():
            raise ValueError("only IMAGE_TOKEN_INDEX may be negative in input_ids")
        sentinel_counts = (input_ids == IMAGE_TOKEN_INDEX).sum(dim=1)
        if not torch.all(sentinel_counts == 1):
            raise ValueError("each sample must contain exactly one IMAGE_TOKEN_INDEX")
        if (input_ids.masked_select(input_ids != IMAGE_TOKEN_INDEX) >= self.llm.embed_tokens.num_embeddings).any():
            raise ValueError("text token id exceeds vocabulary")

        packed_embeds, packed_masks, packed_labels = [], [], []
        n_patch = image_features.shape[1]
        for i in range(b):
            pos = int(torch.where(input_ids[i] == IMAGE_TOKEN_INDEX)[0].item())
            before = self.llm.embed_tokens(input_ids[i, :pos])
            after = self.llm.embed_tokens(input_ids[i, pos + 1:])
            packed_embeds.append(torch.cat((before, image_features[i], after), dim=0))
            packed_masks.append(torch.cat((attention_mask[i, :pos],
                                           torch.ones(n_patch, device=input_ids.device, dtype=attention_mask.dtype),
                                           attention_mask[i, pos + 1:]), dim=0))
            if labels is not None:
                packed_labels.append(torch.cat((labels[i, :pos],
                                                 torch.full((n_patch,), IGNORE_INDEX, device=labels.device, dtype=labels.dtype),
                                                 labels[i, pos + 1:]), dim=0))

        max_len = max(x.shape[0] for x in packed_embeds)
        d = image_features.shape[-1]
        embeds = image_features.new_zeros(b, max_len, d)
        mask = torch.zeros(b, max_len, dtype=attention_mask.dtype, device=input_ids.device)
        out_labels = (torch.full((b, max_len), IGNORE_INDEX, dtype=labels.dtype, device=labels.device)
                      if labels is not None else None)
        for i, (e, m) in enumerate(zip(packed_embeds, packed_masks)):
            embeds[i, :e.shape[0]] = e
            mask[i, :m.shape[0]] = m
            if out_labels is not None:
                out_labels[i, :e.shape[0]] = packed_labels[i]
        return embeds, mask, out_labels

    def forward(self, input_ids, pixel_values, attention_mask=None, labels=None):
        image_features = self.encode_images(pixel_values)
        embeds, mask, packed_labels = self.pack_multimodal_inputs(input_ids, image_features, attention_mask, labels)
        logits = self.llm(inputs_embeds=embeds, attention_mask=mask)
        loss = None
        if packed_labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), packed_labels[:, 1:].reshape(-1),
                                   ignore_index=IGNORE_INDEX)
        return {"loss": loss, "logits": logits, "attention_mask": mask, "labels": packed_labels, "inputs_embeds": embeds}

    @torch.no_grad()
    def generate(self, input_ids, pixel_values, attention_mask=None, max_new_tokens=4):
        self.eval()
        image_features = self.encode_images(pixel_values)
        embeds, mask, _ = self.pack_multimodal_inputs(input_ids, image_features, attention_mask)
        # Decode each row separately after removing right padding.  This keeps a
        # shorter prompt's new token adjacent to its final valid prompt token.
        rows = []
        for i in range(embeds.size(0)):
            row_embeds = embeds[i:i + 1, mask[i].bool()]
            row_mask = torch.ones(1, row_embeds.size(1), device=mask.device, dtype=mask.dtype)
            tokens = []
            for _ in range(max_new_tokens):
                logits = self.llm(inputs_embeds=row_embeds, attention_mask=row_mask)
                next_id = logits[:, -1].argmax(dim=-1)
                tokens.append(next_id)
                row_embeds = torch.cat((row_embeds, self.llm.embed_tokens(next_id).unsqueeze(1)), dim=1)
                row_mask = torch.cat((row_mask, torch.ones(1, 1, device=mask.device, dtype=mask.dtype)), dim=1)
            rows.append(torch.cat(tokens))
        return torch.stack(rows)


def build_toy_llava(vocab_size=64) -> LlavaForConditionalGeneration:
    vision = MockVisionEncoder(image_size=8, patch_size=4, hidden_size=24)
    llm = MockDecoderLM(vocab_size=vocab_size, hidden_size=32, max_positions=64)
    return LlavaForConditionalGeneration(vision, llm, LlavaProjector(24, 32), pad_token_id=0)
