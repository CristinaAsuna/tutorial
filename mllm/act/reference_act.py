"""Action Chunking with Transformers (ACT), reduced to CPU-teachable tensors.

This reference preserves ACT's algorithmic contract: predict a future action
chunk from multi-view images and proprioception; use a training-only CVAE style
encoder; at inference decode with z=0; and combine overlapping chunks with the
paper's temporal ensemble.  It is not an ALOHA hardware driver.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def sample_action_chunks(actions: Tensor, chunk_size: int) -> tuple[Tensor, Tensor]:
    """Turn ``(B,T,A)`` actions into future ``(B,T,K,A)`` chunks and masks."""
    if actions.ndim != 3 or chunk_size <= 0:
        raise ValueError("actions must be (B,T,A) and chunk_size must be positive")
    batch, steps, action_dim = actions.shape
    if steps <= 0:
        raise ValueError("trajectory must contain at least one action")
    start = torch.arange(steps, device=actions.device)[:, None]
    offset = torch.arange(chunk_size, device=actions.device)[None, :]
    index = start + offset
    mask = index < steps
    chunks = actions[:, index.clamp_max(steps - 1)]
    return chunks, mask.expand(batch, -1, -1)


def masked_l1(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Mean L1 over valid action timesteps only."""
    if prediction.shape != target.shape or mask.shape != prediction.shape[:2]:
        raise ValueError("prediction/target must be (B,K,A) and mask must be (B,K)")
    if not mask.bool().any():
        raise ValueError("at least one action target must be valid")
    valid = mask.bool().unsqueeze(-1).expand_as(prediction)
    return F.l1_loss(prediction[valid], target[valid])


def kl_to_standard_normal(mean: Tensor, logvar: Tensor) -> Tensor:
    if mean.shape != logvar.shape:
        raise ValueError("mean and logvar must have matching shape")
    return 0.5 * (mean.square() + logvar.exp() - 1.0 - logvar).sum(dim=-1).mean()


@dataclass
class TemporalEnsembler:
    """Combine predictions for the current action from overlapping chunks.

    Call ``add`` once at every observation.  The oldest chunk is indexed first
    and receives paper-style weight ``exp(-decay * i)``.  Keeping only K chunks
    is sufficient: a chunk older than K cannot predict today's action.
    """
    chunk_size: int
    action_dim: int
    decay: float = 0.01
    _history: list[Tensor] = field(default_factory=list, init=False, repr=False)

    def add(self, chunk: Tensor) -> Tensor:
        if chunk.ndim != 2 or chunk.shape != (self.chunk_size, self.action_dim):
            raise ValueError("chunk must be (chunk_size, action_dim)")
        self._history.append(chunk)
        self._history = self._history[-self.chunk_size:]
        n = len(self._history)
        # Chunk i was predicted i observations ago, so its proposal for now is
        # at offset n-1-i.  All proposals therefore describe the same time.
        proposals = torch.stack([chunk_i[n - 1 - i] for i, chunk_i in enumerate(self._history)])
        weights = torch.exp(-self.decay * torch.arange(n, device=chunk.device, dtype=chunk.dtype))
        return (proposals * weights[:, None]).sum(dim=0) / weights.sum()


class StyleEncoder(nn.Module):
    """Training-only BERT-like encoder q(z | current qpos, future chunk)."""
    def __init__(self, qpos_dim: int, action_dim: int, chunk_size: int, dim: int, latent_dim: int, heads: int):
        super().__init__()
        self.qpos_proj = nn.Linear(qpos_dim, dim)
        self.action_proj = nn.Linear(action_dim, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, chunk_size + 2, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, batch_first=True, dropout=0.0, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.mean = nn.Linear(dim, latent_dim)
        self.logvar = nn.Linear(dim, latent_dim)
        nn.init.normal_(self.cls, std=0.02); nn.init.normal_(self.pos, std=0.02)

    def forward(self, qpos: Tensor, actions: Tensor, action_mask: Tensor) -> tuple[Tensor, Tensor]:
        if actions.ndim != 3 or action_mask.shape != actions.shape[:2]:
            raise ValueError("actions must be (B,K,A) and action_mask (B,K)")
        batch = qpos.shape[0]
        tokens = torch.cat((self.cls.expand(batch, -1, -1), self.qpos_proj(qpos).unsqueeze(1), self.action_proj(actions)), dim=1)
        valid = torch.cat((torch.ones(batch, 2, dtype=torch.bool, device=qpos.device), action_mask.bool()), dim=1)
        output = self.encoder(tokens + self.pos[:, :tokens.shape[1]], src_key_padding_mask=~valid)
        return self.mean(output[:, 0]), self.logvar(output[:, 0])


class ACTPolicy(nn.Module):
    """Multi-view visual conditional VAE that emits an absolute joint chunk."""
    def __init__(self, *, qpos_dim: int = 4, action_dim: int = 4, chunk_size: int = 5, image_size: int = 8,
                 dim: int = 32, latent_dim: int = 8, heads: int = 4, max_cameras: int = 4):
        super().__init__()
        if image_size % 4 or chunk_size <= 0 or max_cameras <= 0:
            raise ValueError("image_size must be divisible by 4; chunk_size and max_cameras positive")
        self.qpos_dim, self.action_dim, self.chunk_size, self.image_size = qpos_dim, action_dim, chunk_size, image_size
        self.visual = nn.Conv2d(3, dim, kernel_size=4, stride=4)
        patches = (image_size // 4) ** 2
        self.spatial_pos = nn.Parameter(torch.zeros(1, patches, dim))
        self.camera_embed = nn.Embedding(max_cameras, dim)
        self.qpos_proj, self.z_proj = nn.Linear(qpos_dim, dim), nn.Linear(latent_dim, dim)
        self.style_encoder = StyleEncoder(qpos_dim, action_dim, chunk_size, dim, latent_dim, heads)
        memory_layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, batch_first=True, dropout=0.0, activation="gelu")
        self.memory_encoder = nn.TransformerEncoder(memory_layer, num_layers=1)
        self.action_queries = nn.Parameter(torch.zeros(1, chunk_size, dim))
        decoder_layer = nn.TransformerDecoderLayer(dim, heads, dim * 2, batch_first=True, dropout=0.0, activation="gelu")
        self.action_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.action_head = nn.Linear(dim, action_dim)
        nn.init.normal_(self.spatial_pos, std=0.02)
        nn.init.normal_(self.action_queries, std=0.02)

    def _encode_observation(self, images: Tensor, qpos: Tensor, z: Tensor) -> Tensor:
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError("images must be (B,num_cameras,3,H,W)")
        batch, cameras, _, height, width = images.shape
        if (height, width) != (self.image_size, self.image_size) or qpos.shape != (batch, self.qpos_dim):
            raise ValueError("invalid image resolution or qpos shape")
        if cameras > self.camera_embed.num_embeddings:
            raise ValueError("too many cameras for configured camera embeddings")
        features = self.visual(images.flatten(0, 1)).flatten(2).transpose(1, 2)
        patches = features.shape[1]
        features = features.reshape(batch, cameras, patches, -1)
        features = features + self.spatial_pos[:, None] + self.camera_embed(torch.arange(cameras, device=images.device))[None, :, None]
        features = features.reshape(batch, cameras * patches, -1)
        tokens = torch.cat((features, self.qpos_proj(qpos).unsqueeze(1), self.z_proj(z).unsqueeze(1)), dim=1)
        return self.memory_encoder(tokens)

    def _decode(self, images: Tensor, qpos: Tensor, z: Tensor) -> Tensor:
        memory = self._encode_observation(images, qpos, z)
        queries = self.action_queries.expand(images.shape[0], -1, -1)
        return self.action_head(self.action_decoder(queries, memory))

    def forward(self, images: Tensor, qpos: Tensor, action_chunks: Tensor | None = None,
                action_mask: Tensor | None = None, *, beta: float = 10.0) -> dict[str, Tensor | None]:
        """Use posterior z in training; set z=0 for action-chunk inference."""
        if action_chunks is None:
            z = torch.zeros(qpos.shape[0], self.z_proj.in_features, device=qpos.device, dtype=qpos.dtype)
            return {"actions": self._decode(images, qpos, z), "loss": None, "reconstruction_loss": None, "kl_loss": None}
        if action_chunks.shape != (qpos.shape[0], self.chunk_size, self.action_dim) or action_mask is None:
            raise ValueError("action_chunks must be (B,chunk_size,action_dim) with action_mask")
        mean, logvar = self.style_encoder(qpos, action_chunks, action_mask)
        z = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
        prediction = self._decode(images, qpos, z)
        reconstruction = masked_l1(prediction, action_chunks, action_mask)
        kl = kl_to_standard_normal(mean, logvar)
        return {"actions": prediction, "loss": reconstruction + beta * kl,
                "reconstruction_loss": reconstruction, "kl_loss": kl, "mean": mean, "logvar": logvar}

    @torch.no_grad()
    def predict_chunk(self, images: Tensor, qpos: Tensor) -> Tensor:
        self.eval()
        return self(images, qpos)["actions"]


def build_toy_act() -> ACTPolicy:
    return ACTPolicy()
