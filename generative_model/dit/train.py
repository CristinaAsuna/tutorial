"""Train the same DiT architecture under one selected generative objective.

Examples (run from project root):
    python -m dit.train --objective ddpm
    python -m dit.train --objective flow_matching
    python -m dit.train --objective vp_sde
"""

import argparse
import sys
from pathlib import Path

from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dit.pipeline import build_pipeline
from utils.data import NpzImageDataset, build_image_transform
from utils.dit import DiT, DiTConfig
from utils.trainers import (
    DDPMTrainConfig,
    DDPMTrainer,
    FlowMatchingTrainConfig,
    FlowMatchingTrainer,
    ScoreSDETrainConfig,
    ScoreSDETrainer,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", choices=["ddpm", "flow_matching", "vp_sde"], required=True)
    parser.add_argument("--data", default=r"D:\Datasets\cifar10\train.npz")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=384)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=6)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(__file__).with_name("checkpoints") / args.objective

    dataset = NpzImageDataset(args.data, build_image_transform(args.image_size, value_range="tanh"))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = DiT(DiTConfig(
        in_channels=3,
        image_size=args.image_size,
        patch_size=args.patch_size,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
    ))
    pipeline = build_pipeline(args.objective)

    if args.objective == "ddpm":
        trainer = DDPMTrainer(model, pipeline, DDPMTrainConfig(output_dir=str(output_dir), epochs=args.epochs))
    elif args.objective == "flow_matching":
        trainer = FlowMatchingTrainer(model, pipeline, FlowMatchingTrainConfig(
            data_root=args.data, output_dir=str(output_dir), image_size=args.image_size,
            batch_size=args.batch_size, epochs=args.epochs,
        ))
    else:
        trainer = ScoreSDETrainer(model, pipeline, ScoreSDETrainConfig(output_dir=str(output_dir), epochs=args.epochs))

    trainer.fit(loader)


if __name__ == "__main__":
    main()
