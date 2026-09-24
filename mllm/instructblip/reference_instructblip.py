"""InstructBLIP's instruction-aware Q-Former on CPU-sized toy tensors.

The important delta from BLIP-2 Stage 2 is that learnable visual queries and
instruction tokens self-attend before queries cross-attend to image patches.
Therefore the same image produces different visual soft prompts for different
instructions.  This is a teaching model, not a checkpoint loader.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from edu_core.attention import MultiHeadAttention
from edu_core.training import freeze_and_keep_eval, set_requires_grad

IGNORE_INDEX = -100


class MockVisionEncoder(nn.Module):
    def __init__(self, image_size: int = 8, patch_size: int = 4, vision_dim: int = 24):
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.proj = nn.Conv2d(3, vision_dim, patch_size, patch_size)

    def forward(self, pixel_values: Tensor) -> Tensor:
        if pixel_values.ndim != 4 or pixel_values.shape[1] != 3:
            raise ValueError("pixel_values must be (B,3,H,W)")
        if tuple(pixel_values.shape[-2:]) != (self.image_size, self.image_size):
            raise ValueError(f"expected {self.image_size}x{self.image_size} images")
        return self.proj(pixel_values).flatten(2).transpose(1, 2)


class InstructionQFormerLayer(nn.Module):
    """Text and queries self-attend; only queries cross-attend to vision."""
    def __init__(self, dim: int, num_heads: int, vision_dim: int, mlp_ratio: float = 2.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = MultiHeadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = MultiHeadAttention(dim, num_heads, kv_dim=vision_dim)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, tokens: Tensor, num_queries: int, vision_features: Tensor, valid_tokens: Tensor) -> Tensor:
        tokens = tokens + self.self_attn(self.norm1(tokens), key_padding_mask=valid_tokens)
        queries = tokens[:, :num_queries]
        queries = queries + self.cross_attn(self.norm2(queries), vision_features)
        tokens = torch.cat((queries, tokens[:, num_queries:]), dim=1)
        return tokens + self.mlp(self.norm3(tokens))


class InstructionAwareQFormer(nn.Module):
    """Produce visual queries conditioned on an instruction-token sequence."""
    def __init__(self, *, instruction_vocab_size: int = 40, vision_dim: int = 24, dim: int = 32,
                 num_queries: int = 4, depth: int = 2, num_heads: int = 4, max_instruction_length: int = 16):
        super().__init__()
        if num_queries <= 0:
            raise ValueError("num_queries must be positive")
        self.dim, self.num_queries, self.max_instruction_length = dim, num_queries, max_instruction_length
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        self.word_embeddings = nn.Embedding(instruction_vocab_size, dim)
        self.position_embeddings = nn.Embedding(max_instruction_length, dim)
        self.layers = nn.ModuleList([InstructionQFormerLayer(dim, num_heads, vision_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, vision_features: Tensor, instruction_ids: Tensor, instruction_attention_mask: Optional[Tensor] = None) -> Tensor:
        if vision_features.ndim != 3 or instruction_ids.ndim != 2 or vision_features.shape[0] != instruction_ids.shape[0]:
            raise ValueError("vision_features (B,N,Dv) and instruction_ids (B,L) need matching batches")
        b, length = instruction_ids.shape
        if length == 0 or length > self.max_instruction_length:
            raise ValueError("instruction length must be in [1, max_instruction_length]")
        if instruction_attention_mask is None:
            instruction_attention_mask = torch.ones_like(instruction_ids, dtype=torch.bool)
        if instruction_attention_mask.shape != instruction_ids.shape or not instruction_attention_mask.bool().any(dim=1).all():
            raise ValueError("every instruction needs at least one valid token and a matching mask")
        if (instruction_ids < 0).any() or (instruction_ids >= self.word_embeddings.num_embeddings).any():
            raise ValueError("instruction token id is outside the Q-Former vocabulary")
        positions = torch.arange(length, device=instruction_ids.device).unsqueeze(0)
        text = self.word_embeddings(instruction_ids) + self.position_embeddings(positions)
        queries = self.query_tokens.expand(b, -1, -1)
        tokens = torch.cat((queries, text), dim=1)
        valid = torch.cat((torch.ones(b, self.num_queries, dtype=torch.bool, device=tokens.device), instruction_attention_mask.bool()), dim=1)
        for layer in self.layers:
            tokens = layer(tokens, self.num_queries, vision_features, valid)
        return self.norm(tokens[:, :self.num_queries])


class MockDecoderLM(nn.Module):
    """Small frozen causal decoder accepting visual soft prompts as embeddings."""
    def __init__(self, vocab_size: int = 48, dim: int = 32, max_positions: int = 64, num_heads: int = 4):
        super().__init__()
        self.dim, self.max_positions = dim, max_positions
        self.embed_tokens = nn.Embedding(vocab_size, dim)
        self.position_embeddings = nn.Embedding(max_positions, dim)
        layer = nn.TransformerEncoderLayer(dim, num_heads, dim * 2, batch_first=True, dropout=0.0, activation="gelu")
        self.decoder = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, inputs_embeds: Tensor, attention_mask: Tensor) -> Tensor:
        if inputs_embeds.ndim != 3 or attention_mask.shape != inputs_embeds.shape[:2]:
            raise ValueError("inputs_embeds and attention_mask must be (B,L,D) and (B,L)")
        b, length, dim = inputs_embeds.shape
        if dim != self.dim or length > self.max_positions:
            raise ValueError("invalid decoder hidden size or sequence length")
        positions = torch.arange(length, device=inputs_embeds.device).unsqueeze(0)
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=inputs_embeds.device), diagonal=1)
        x = self.decoder(inputs_embeds + self.position_embeddings(positions), mask=causal,
                         src_key_padding_mask=~attention_mask.bool())
        return self.lm_head(self.norm(x))


class InstructBlipForConditionalGeneration(nn.Module):
    """Frozen visual/LLM experts with a trainable instruction-aware Q-Former."""
    def __init__(self, vision_encoder: MockVisionEncoder, qformer: InstructionAwareQFormer, llm: MockDecoderLM):
        super().__init__()
        if qformer.dim != llm.dim:
            raise ValueError("Q-Former and LLM dimensions must agree in this toy model")
        self.vision_encoder, self.qformer, self.llm = vision_encoder, qformer, llm
        self.llm_proj = nn.Linear(qformer.dim, llm.dim)
        self.freeze_backbones()

    def freeze_backbones(self) -> None:
        freeze_and_keep_eval(self.vision_encoder)
        freeze_and_keep_eval(self.llm)

    def train(self, mode: bool = True):
        super().train(mode)
        self.vision_encoder.eval(); self.llm.eval()
        return self

    def encode_instruction_aware_queries(self, pixel_values: Tensor, instruction_ids: Tensor,
                                         instruction_attention_mask: Optional[Tensor] = None) -> Tensor:
        with torch.no_grad():
            vision_features = self.vision_encoder(pixel_values)
        return self.qformer(vision_features, instruction_ids, instruction_attention_mask)

    def _validate_llm_inputs(self, prompt_ids: Tensor, answer_ids: Tensor, prompt_mask: Tensor, answer_mask: Tensor) -> None:
        if prompt_ids.ndim != 2 or answer_ids.ndim != 2 or prompt_ids.shape[0] != answer_ids.shape[0]:
            raise ValueError("prompt_ids and answer_ids must be matching (B,L) tensors")
        if prompt_mask.shape != prompt_ids.shape or answer_mask.shape != answer_ids.shape:
            raise ValueError("prompt/answer masks must match their ids")
        if not prompt_mask.bool().any(dim=1).all() or not answer_mask.bool().any(dim=1).all():
            raise ValueError("every sample needs a valid prompt and answer token")
        vocab = self.llm.embed_tokens.num_embeddings
        if (prompt_ids < 0).any() or (answer_ids < 0).any() or (prompt_ids >= vocab).any() or (answer_ids >= vocab).any():
            raise ValueError("LLM token id is outside vocabulary")

    def forward(self, pixel_values: Tensor, instruction_ids: Tensor, llm_prompt_ids: Tensor, answer_ids: Tensor, *,
                instruction_attention_mask: Optional[Tensor] = None, llm_prompt_attention_mask: Optional[Tensor] = None,
                answer_attention_mask: Optional[Tensor] = None) -> dict[str, Tensor]:
        instruction_attention_mask = torch.ones_like(instruction_ids, dtype=torch.bool) if instruction_attention_mask is None else instruction_attention_mask.bool()
        llm_prompt_attention_mask = torch.ones_like(llm_prompt_ids, dtype=torch.bool) if llm_prompt_attention_mask is None else llm_prompt_attention_mask.bool()
        answer_attention_mask = torch.ones_like(answer_ids, dtype=torch.bool) if answer_attention_mask is None else answer_attention_mask.bool()
        self._validate_llm_inputs(llm_prompt_ids, answer_ids, llm_prompt_attention_mask, answer_attention_mask)
        queries = self.encode_instruction_aware_queries(pixel_values, instruction_ids, instruction_attention_mask)
        visual_prefix = self.llm_proj(queries)
        rows, labels, masks = [], [], []
        for row in range(pixel_values.shape[0]):
            prompt = self.llm.embed_tokens(llm_prompt_ids[row, llm_prompt_attention_mask[row]])
            # Drop answer padding before causal packing: otherwise a padded
            # embedding could become a predecessor of a later real answer.
            valid_answer_ids = answer_ids[row, answer_attention_mask[row]]
            answer = self.llm.embed_tokens(valid_answer_ids)
            rows.append(torch.cat((visual_prefix[row], prompt, answer), dim=0))
            labels.append(torch.cat((torch.full((visual_prefix.shape[1] + prompt.shape[0],), IGNORE_INDEX, device=pixel_values.device, dtype=torch.long),
                                     valid_answer_ids)))
            masks.append(torch.ones(rows[-1].shape[0], dtype=torch.bool, device=pixel_values.device))
        embeds = nn.utils.rnn.pad_sequence(rows, batch_first=True)
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        mask = nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=False)
        logits = self.llm(embeds, mask)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=IGNORE_INDEX)
        return {"loss": loss, "logits": logits, "labels": labels, "visual_prefix": visual_prefix}

    @torch.no_grad()
    def generate(self, pixel_values: Tensor, instruction_ids: Tensor, llm_prompt_ids: Tensor, *,
                 instruction_attention_mask: Optional[Tensor] = None, llm_prompt_attention_mask: Optional[Tensor] = None,
                 max_new_tokens: int = 4) -> Tensor:
        self.eval()
        instruction_attention_mask = torch.ones_like(instruction_ids, dtype=torch.bool) if instruction_attention_mask is None else instruction_attention_mask.bool()
        llm_prompt_attention_mask = torch.ones_like(llm_prompt_ids, dtype=torch.bool) if llm_prompt_attention_mask is None else llm_prompt_attention_mask.bool()
        dummy_answers = torch.ones(llm_prompt_ids.shape[0], 1, dtype=torch.long, device=llm_prompt_ids.device)
        self._validate_llm_inputs(llm_prompt_ids, dummy_answers, llm_prompt_attention_mask, torch.ones_like(dummy_answers, dtype=torch.bool))
        prefix = self.llm_proj(self.encode_instruction_aware_queries(pixel_values, instruction_ids, instruction_attention_mask))
        outputs = []
        for row in range(pixel_values.shape[0]):
            embeds = torch.cat((prefix[row:row + 1], self.llm.embed_tokens(llm_prompt_ids[row, llm_prompt_attention_mask[row]]).unsqueeze(0)), dim=1)
            mask = torch.ones(1, embeds.shape[1], dtype=torch.bool, device=embeds.device)
            generated = []
            for _ in range(max_new_tokens):
                next_id = self.llm(embeds, mask)[:, -1].argmax(dim=-1)
                generated.append(next_id)
                embeds = torch.cat((embeds, self.llm.embed_tokens(next_id).unsqueeze(1)), dim=1)
                mask = torch.cat((mask, torch.ones(1, 1, dtype=torch.bool, device=mask.device)), dim=1)
            outputs.append(torch.cat(generated))
        return torch.stack(outputs)


def build_toy_instructblip() -> InstructBlipForConditionalGeneration:
    vision = MockVisionEncoder()
    qformer = InstructionAwareQFormer()
    llm = MockDecoderLM()
    return InstructBlipForConditionalGeneration(vision, qformer, llm)
