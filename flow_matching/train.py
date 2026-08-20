"""Train an unconditional linear Flow Matching model on local CIFAR-10 NPZ data."""

import sys
from pathlib import Path

from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .model import FlowMatchingUNet
    from .pipeline import FlowMatchingPipeline
except ImportError:  # Supports `python train.py` inside flow_matching.
    from flow_matching.model import FlowMatchingUNet
    from flow_matching.pipeline import FlowMatchingPipeline

from utils.data import NpzImageDataset, build_image_transform
from utils.trainers import FlowMatchingTrainConfig, FlowMatchingTrainer


def main():
    config = FlowMatchingTrainConfig(
        data_root=r"D:\Datasets\cifar10\train.npz",
        output_dir=str(Path(__file__).with_name("checkpoints")),
        image_size=32,
        batch_size=64,
        epochs=20,
        num_workers=4,
        learning_rate=2e-4,
        sample_steps=100,
        sample_count=16,
        time_scale=1000.0,
    )

    dataset = NpzImageDataset(
        config.data_root,
        build_image_transform(config.image_size, value_range="tanh"),
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
    )

    model = FlowMatchingUNet(
        in_channels=3,
        base_channels=64,
        time_dim=256,
    )
    pipeline = FlowMatchingPipeline(time_scale=config.time_scale)
    FlowMatchingTrainer(model, pipeline, config).fit(loader)


if __name__ == "__main__":
    main()
