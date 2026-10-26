"""Provided module initialization and batch engineering.
The learning path injects every defining computation through ops. Read this file for
attribute names and layer shapes; no student TODO needs to design a network constructor.
"""
import torch
from torch import nn
from torch.nn import functional as F
from edu_core import SelfAttentionBlock
IGNORE_INDEX = -100


class TinyVision(nn.Module):
    """Peripheral backbone: 8x8 RGB -> four 16-D patch features; not DINOv2/SigLIP."""
    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(3, 16, 4, 4)
        self.pos = nn.Parameter(torch.randn(1, 4, 16) * .02)
        self.block = SelfAttentionBlock(16, 2)

    def forward(self, images):
        return self.block(self.patch(images).flatten(2).transpose(1, 2) + self.pos)



class TinyLanguage(nn.Module):
    """Provided causal LM; learned positions, no RoPE/GQA/KV-cache compatibility."""
    def __init__(self, vocab_size=320, dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.position = nn.Embedding(96, dim)
        self.blocks = nn.ModuleList([SelfAttentionBlock(dim, 4) for _ in range(2)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x, mask):
        x = x + self.position(torch.arange(x.shape[1], device=x.device))[None]
        for block in self.blocks:
            x = block(x, key_padding_mask=mask, causal=True)
        return self.head(self.norm(x))



class OpenVLAPolicy(nn.Module):
    def __init__(self, ops=None, language=None):
        super().__init__()
        if ops is None:
            from reference_openvla import reference_ops
            ops = reference_ops()
        self.ops = ops
        self.vocab_size, self.action_dim = (320 if language is None else language.vocab_size), 7
        language_dim = 32 if language is None else language.hidden_size
        self.first, self.second = TinyVision(), TinyVision()
        self.projector = nn.Sequential(nn.Linear(32, language_dim), nn.GELU(), nn.Linear(language_dim, language_dim))
        self.language = TinyLanguage(self.vocab_size) if language is None else language

    def train(self, mode=True):
        super().train(mode)
        if not any(p.requires_grad for p in self.language.parameters()):
            self.language.eval()
        return self

    def encode_images(self, images):
        return self.ops.fuse_features(self.first(images), self.second(images), self.projector)

    def pack(self, rows, visual, labels=None):
        # Variable-length batch engineering supplied; only single-row insertion is a lesson.
        packed = [self.ops.pack_single(self.language.embedding(row), image,
                  row.new_full(row.shape, IGNORE_INDEX) if labels is None else label)
                  for row, image, label in zip(rows, visual, [None]*len(rows) if labels is None else labels)]
        width = max(len(x) for x, _ in packed)
        embeds = torch.stack([F.pad(x, (0, 0, 0, width-len(x))) for x, _ in packed])
        targets = torch.stack([F.pad(y, (0, width-len(y)), value=IGNORE_INDEX) for _, y in packed])
        lengths = torch.tensor([len(x) for x, _ in packed], device=visual.device)
        mask = torch.arange(width, device=visual.device)[None] < lengths[:, None]
        return embeds, mask, targets

    def forward(self, images, prompts, actions, stats):
        ids = self.ops.encode_actions(self.ops.normalize_actions(actions, stats), self.vocab_size)
        rows = [torch.cat((p, a, p.new_tensor([2]))) for p, a in zip(prompts, ids)]
        labels = [self.ops.action_labels(row, self.action_dim) for row in rows]
        embeds, mask, targets = self.pack(rows, self.encode_images(images), labels)
        return self.ops.shifted_loss(self.language(embeds, mask), targets)

    @torch.no_grad()
    def predict_action(self, images, prompts, stats):
        self.eval()
        return self.ops.predict_action(self, images, prompts, stats)

