"""DDPM CIFAR-10 training entry point.

This becomes runnable after you implement DDPMPipeline TODOs.
"""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .model import DDPMUNet
    from .pipeline import DDPMPipeline
except ImportError:
    from diffusion.model import DDPMUNet
    from diffusion.pipeline import DDPMPipeline

from utils.data import NpzImageDataset, build_image_transform
from utils.trainers import DDPMTrainConfig, DDPMTrainer


def main():
    config = DDPMTrainConfig(
        output_dir=str(Path(__file__).with_name("checkpoints")),
        epochs=20,
        learning_rate=2e-4,
        sample_count=16,
        sample_every=5,
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

    model = DDPMUNet(in_channels=3, base_channels=64, time_dim=256)
    pipeline = DDPMPipeline(num_train_steps=1000, beta_start=1e-4, beta_end=2e-2)
    DDPMTrainer(model, pipeline, config).fit(loader)


if __name__ == "__main__":
    main()
