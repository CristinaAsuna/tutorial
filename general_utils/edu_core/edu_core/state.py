"""Portable checkpoint payload helpers without a trainer-specific schema."""
from dataclasses import asdict, is_dataclass
from pathlib import Path
import torch


def save_checkpoint(path, *, model, step: int, optimizers=None, config=None, metadata=None) -> None:
    payload = {"model": model.state_dict(), "step": step, "metadata": metadata or {}}
    if optimizers:
        payload["optimizers"] = {name: optimizer.state_dict() for name, optimizer in optimizers.items()}
    if config is not None:
        payload["config"] = asdict(config) if is_dataclass(config) else dict(config)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path, *, model, device, optimizers=None, strict=True):
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"], strict=strict)
    for name, optimizer in (optimizers or {}).items():
        if name in payload.get("optimizers", {}):
            optimizer.load_state_dict(payload["optimizers"][name])
    return payload
