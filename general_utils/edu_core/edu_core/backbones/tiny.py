"""Offline reference backend with the same token/mask/KV contracts as pretrained backends."""
import torch
from torch import nn
from .layers import DecoderLayer, RMSNorm, projected_attention
from .types import DecoderSpec, LanguageOutput, VisionFeatures, valid_positions


class TinyBackbone(nn.Module):
    capabilities = frozenset({"vision", "text", "generation", "block_mask", "layer_kv"})
    def __init__(self, *, hidden_size=64, num_layers=2, vocab_size=128):
        super().__init__()
        self.spec = DecoderSpec(hidden_size, num_layers, 4, 2, 16, vocab_size)
        self.vision_dim = 24
        self.vision = nn.Conv2d(3, self.vision_dim, 4, 4)
        self.connector = nn.Linear(self.vision_dim, hidden_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(self.spec) for _ in range(num_layers)])
        self.norm, self.head = RMSNorm(hidden_size), nn.Linear(hidden_size, vocab_size, bias=False)

    def embed_text(self, ids):
        return self.embedding(ids)

    def encode_vision(self, images, *, projected=False):
        x = self.vision(images)
        grid = tuple(x.shape[-2:])
        tokens = x.flatten(2).transpose(1, 2)
        if projected:
            tokens = self.connector(tokens)
        return VisionFeatures(tokens, torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device), grid)

    def language(self, embeddings, valid_mask, *, allow_mask=None, positions=None, use_cache=False, output_logits=True):
        length = embeddings.shape[1]
        if allow_mask is None:
            allow_mask = torch.ones(length, length, dtype=torch.bool, device=embeddings.device).tril()[None]
            allow_mask = allow_mask & valid_mask[:, :, None] & valid_mask[:, None, :]
        if positions is None:
            positions = valid_positions(valid_mask)
        x, cache = embeddings, []
        for layer in self.layers:
            q, k, v = layer.project(x, positions)
            x = layer.finish(x, projected_attention(q, k, v, allow_mask))
            if use_cache:
                cache.append((k, v))
        hidden = self.norm(x)
        return LanguageOutput(hidden, self.head(hidden) if output_logits else None, tuple(cache))

    @torch.no_grad()
    def generate(self, ids, max_new_tokens=4):
        self.eval()
        for _ in range(max_new_tokens):
            valid = torch.ones_like(ids, dtype=torch.bool)
            logits = self.language(self.embed_text(ids), valid).logits
            ids = torch.cat((ids, logits[:, -1].argmax(-1, keepdim=True)), -1)
        return ids
