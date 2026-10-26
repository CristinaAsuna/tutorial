"""Original OpenVLA mechanisms; tiny randomly initialized backbones, not pretrained VLA."""
from __future__ import annotations

from types import SimpleNamespace
import torch
from torch.nn import functional as F

IGNORE_INDEX = -100


def fit_action_stats(actions, mask):
    """Fit training-only (N,A) quantiles; mask (A,) selects normalized dimensions."""
    return {"q01": torch.quantile(actions, .01, dim=0),
            "q99": torch.quantile(actions, .99, dim=0), "mask": mask.bool()}


def normalize_actions(actions, stats):
    low, high = stats["q01"], stats["q99"]
    scaled = (2 * (actions - low) / (high - low).clamp_min(1e-6) - 1).clamp(-1, 1)
    return torch.where(stats["mask"], scaled, actions)


def unnormalize_actions(actions, stats):
    low, high = stats["q01"], stats["q99"]
    return torch.where(stats["mask"], (actions + 1) * .5 * (high - low) + low, actions)


def encode_actions(actions, vocab_size=320, n_bins=256):
    """Match official np.digitize: 256 edges, 255 centers, token V - index."""
    # NumPy's official linspace uses float64. Float32 boundaries change IDs near interior edges.
    edges = torch.arange(n_bins, device=actions.device, dtype=torch.float64) * (2/(n_bins-1)) - 1
    indices = torch.bucketize(actions.clamp(-1, 1).double().contiguous(), edges, right=True)
    return vocab_size - indices


def decode_actions(ids, vocab_size=320, n_bins=256):
    edges = torch.linspace(-1, 1, n_bins, device=ids.device)
    centers = (edges[:-1] + edges[1:]) / 2
    indices = (vocab_size - ids - 1).clamp(0, n_bins - 2)
    return centers[indices]


def fuse_features(first, second, projector):
    """(B,N,D1)+(B,N,D2) -> (B,N,D); concatenate channels, never patches."""
    return projector(torch.cat((first, second), dim=-1))


def pack_single(text_embeds, visual, labels):
    """Insert visual features after BOS; visual labels are ignored."""
    embeds = torch.cat((text_embeds[:1], visual, text_embeds[1:]))
    expanded = torch.cat((labels[:1], labels.new_full((len(visual),), IGNORE_INDEX), labels[1:]))
    return embeds, expanded


def action_labels(ids, action_dim, predict_stop=True):
    """Training row is BOS + prompt + A action tokens + EOS."""
    labels = ids.clone()
    labels[:-(action_dim + 1)] = IGNORE_INDEX
    if not predict_stop:
        labels[-1] = IGNORE_INDEX
    return labels


def shifted_loss(logits, labels):
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                           labels[:, 1:].reshape(-1), ignore_index=IGNORE_INDEX)


def training_step(model, images, prompts, actions, stats, optimizer):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = model(images, prompts, actions, stats)
    loss.backward()
    optimizer.step()
    return loss.detach()


def generate_action_ids(model, images, prompts):
    """Full-vocabulary greedy generation, fixed A steps; no artificial action-only mask."""
    rows = [p.clone() for p in prompts]
    visual = model.encode_images(images)
    generated = []
    for _ in range(model.action_dim):
        embeds, mask, _ = model.pack(rows, visual)
        logits = model.language(embeds, mask)
        lengths = mask.sum(1) - 1
        next_ids = logits[torch.arange(len(rows), device=images.device), lengths].argmax(-1)
        generated.append(next_ids)
        rows = [torch.cat((row, token[None])) for row, token in zip(rows, next_ids)]
    return torch.stack(generated, dim=1)


def predict_action(model, images, prompts, stats):
    ids = model.ops.generate_action_ids(model, images, prompts)
    # Official decoder clips every ID, including non-action text IDs, to its center range.
    return model.ops.unnormalize_actions(model.ops.decode_actions(ids, model.vocab_size), stats)


def reference_ops():
    names = ("fit_action_stats", "normalize_actions", "unnormalize_actions", "encode_actions",
             "decode_actions", "fuse_features", "pack_single", "action_labels", "shifted_loss",
             "training_step", "generate_action_ids", "predict_action")
    return SimpleNamespace(**{name: globals()[name] for name in names})


from scaffolding import TinyVision, TinyLanguage, OpenVLAPolicy

def build_toy_openvla() -> OpenVLAPolicy:
    return OpenVLAPolicy()
