"""Explicit compatibility wrappers for existing lessons, with disjoint parameter ownership."""
import torch
from torch import nn
from .types import valid_positions


class VisionWithCLS(nn.Module):
    """Raw pretrained patches pooled to 4x4 for CPU, plus a synthetic removable CLS slot.

    SigLIP has no CLS token. Existing LLaVA lessons intentionally strip index 0;
    this wrapper supplies a mean token solely for that legacy contract, not as a
    claim about SigLIP's native architecture. Only the vision tower is registered.
    """
    def __init__(self, backbone, output_grid=4):
        super().__init__()
        self.tower = backbone.model.model.vision_model
        self.image_size = backbone.model.config.vision_config.image_size
        self.output_grid = output_grid
        self.register_buffer("cls_token", torch.zeros(1, 1, backbone.vision_dim))
        self.register_buffer("mean", torch.tensor(backbone.processor.image_processor.image_mean)[None, :, None, None])
        self.register_buffer("std", torch.tensor(backbone.processor.image_processor.image_std)[None, :, None, None])

    def forward(self, images):
        pixels = torch.nn.functional.interpolate(images, (self.image_size, self.image_size), mode="bilinear", align_corners=False)
        tokens = self.tower(((pixels-self.mean)/self.std).to(next(self.tower.parameters()).dtype)).last_hidden_state
        side = round(tokens.shape[1]**.5)
        grid = tokens.transpose(1, 2).reshape(len(tokens), -1, side, side)
        patches = torch.nn.functional.adaptive_avg_pool2d(grid, self.output_grid).flatten(2).transpose(1, 2)
        return torch.cat((patches.mean(1, keepdim=True), patches), 1)


class CausalLanguage(nn.Module):
    """Only language modules; freezing a vision wrapper cannot freeze this decoder."""
    def __init__(self, backbone):
        super().__init__()
        self.decoder = backbone.text_model
        self.head = backbone.model.lm_head
        self.hidden_size, self.vocab_size = backbone.spec.hidden_size, backbone.spec.vocab_size

    @property
    def embed_tokens(self):
        return self.decoder.get_input_embeddings()

    @property
    def embedding(self):
        return self.embed_tokens

    def train(self, mode=True):
        return super().train(mode and any(p.requires_grad for p in self.parameters()))

    def forward(self, input_ids=None, *, inputs_embeds=None, attention_mask=None):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones(inputs_embeds.shape[:2], device=inputs_embeds.device, dtype=torch.bool)
        hidden = self.decoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                              position_ids=valid_positions(attention_mask), use_cache=False).last_hidden_state
        return self.head(hidden)


class EmbeddingLanguage(CausalLanguage):
    """OpenVLA's existing forward(embeddings, mask) call convention."""
    def forward(self, embeddings, mask):
        return super().forward(inputs_embeds=embeddings, attention_mask=mask)
