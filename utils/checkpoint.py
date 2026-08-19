from dataclasses import asdict, is_dataclass

import torch


def save_checkpoint(path, model, epoch, global_step=0, optimizers=None, config=None, extra=None):
    """保存模型、可选优化器状态与配置；适用于任何训练器。"""
    state = {
        "model": model.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }
    if optimizers:
        state["optimizers"] = {name: optimizer.state_dict() for name, optimizer in optimizers.items()}
    if config is not None:
        state["config"] = asdict(config) if is_dataclass(config) else dict(config)
    if extra:
        state["extra"] = extra
    torch.save(state, path)


def read_checkpoint(path, device):
    """读取 checkpoint 字典；用于根据其中的模型配置创建模型。"""
    return torch.load(path, map_location=device, weights_only=False)


def load_checkpoint(path, model, device, optimizers=None, strict=True):
    """加载 checkpoint，并在提供优化器时恢复它们。"""
    checkpoint = read_checkpoint(path, device)
    model.load_state_dict(checkpoint["model"], strict=strict)
    if optimizers and "optimizers" in checkpoint:
        for name, optimizer in optimizers.items():
            if name in checkpoint["optimizers"]:
                optimizer.load_state_dict(checkpoint["optimizers"][name])
    return checkpoint
