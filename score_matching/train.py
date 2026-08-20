"""Train VP score-SDE on local CIFAR-10 NPZ data."""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .model import ScoreMatchingUNet
    from .pipeline import ScoreMatchingPipeline, VPSDE
except ImportError:  # Supports `python train.py` inside score_matching.
    from score_matching.model import ScoreMatchingUNet
    from score_matching.pipeline import ScoreMatchingPipeline, VPSDE

from utils.data import NpzImageDataset, build_image_transform
from utils.trainers import ScoreSDETrainConfig, ScoreSDETrainer


def main():
    config = ScoreSDETrainConfig(
        output_dir=str(Path(__file__).with_name("checkpoints")),
        epochs=20,
        learning_rate=2e-4,
        sample_steps=200,
        sample_count=16,
        sample_every=5,
        time_scale=1000.0,
        beta_min=0.1,
        beta_max=20.0,
        sde_eps=1e-5,
    )
    batch_size = 64
    num_workers = 4

    dataset = NpzImageDataset(
        r"D:\Datasets\cifar10\train.npz",
        build_image_transform(32, value_range="tanh"),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    model = ScoreMatchingUNet(
        in_channels=3,
        base_channels=64,
        time_dim=256,
    )
    sde = VPSDE(
        beta_min=config.beta_min,
        beta_max=config.beta_max,
        eps=config.sde_eps,
    )
    pipeline = ScoreMatchingPipeline(sde, time_scale=config.time_scale)
    ScoreSDETrainer(model, pipeline, config).fit(loader)


if __name__ == "__main__":
    main()
