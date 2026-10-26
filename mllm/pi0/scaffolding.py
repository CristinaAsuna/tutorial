"""Provided initialization/outer composition. All defining calculations go through ops."""
import torch
from torch import nn
from edu_core.backbones import TinyBackbone, require_capabilities
from edu_core.backbones.layers import DecoderLayer, RMSNorm


class Pi0Policy(nn.Module):
    def __init__(self, backbone=None, ops=None, expert_dim=32):
        super().__init__()
        if ops is None:
            from reference_pi0 import reference_ops
            ops = reference_ops()
        self.ops = ops
        self.backbone = TinyBackbone() if backbone is None else backbone
        require_capabilities(self.backbone, "vision", "text", "block_mask", "layer_kv")
        self.expert_dim, self.action_dim, self.horizon = expert_dim, 7, 8
        if expert_dim < 4 or expert_dim % 2:
            raise ValueError("expert_dim must be even and at least 4 for sinusoidal time embeddings")
        self.state_proj = nn.Linear(7, expert_dim)
        self.action_in = nn.Linear(7, expert_dim)
        self.action_time = nn.Sequential(nn.Linear(2*expert_dim, expert_dim), nn.SiLU(), nn.Linear(expert_dim, expert_dim))
        self.expert_layers = nn.ModuleList([DecoderLayer(self.backbone.spec, expert_dim) for _ in range(self.backbone.spec.num_layers)])
        self.expert_norm, self.action_out = RMSNorm(expert_dim), nn.Linear(expert_dim, 7)
        self.register_buffer("state_low", torch.full((7,), -2.))
        self.register_buffer("state_high", torch.full((7,), 2.))
        self.register_buffer("action_low", torch.full((7,), -2.))
        self.register_buffer("action_high", torch.full((7,), 2.))

    def train(self, mode=True):
        super().train(mode)
        if not any(p.requires_grad for p in self.backbone.parameters()):
            self.backbone.eval()
        return self

    def forward(self, batch, noise=None, time=None):
        actions = self.ops.normalize_values(batch["actions"], self.action_low, self.action_high)
        state = self.ops.normalize_values(batch["state"], self.state_low, self.state_high)
        if noise is None:
            noise = torch.randn_like(actions)
        if time is None:
            time = self.ops.sample_time(len(actions), actions.device)
        noisy, target = self.ops.flow_pair(actions, noise, time)
        prefix, valid = self.ops.embed_prefix(self, batch["images"], batch["camera_valid"], batch["text_ids"], batch["text_valid"])
        suffix = self.ops.embed_suffix(self, state, noisy, time)
        # Joint path is available for the inspectable tiny backbone. The pretrained path
        # computes the mathematically equivalent one-way prefix first, retaining gradients
        # when it is trainable; it never replaces a missing student computation.
        if isinstance(self.backbone, TinyBackbone):
            velocity = self.ops.joint_forward(self, prefix, valid, suffix)
        else:
            velocity = self.ops.cached_forward(self, self.ops.cache_prefix(self, prefix, valid), suffix)
        return self.ops.flow_loss(velocity, target)

    @torch.no_grad()
    def predict_action(self, batch, initial_noise, steps=10):
        if steps <= 0:
            raise ValueError("Euler steps must be positive")
        self.eval()
        return self.ops.sample_actions(self, batch, initial_noise, steps)
