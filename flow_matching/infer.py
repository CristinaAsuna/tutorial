"""Generate CIFAR-sized images from a trained Flow Matching checkpoint."""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .model import FlowMatchingUNet
    from .pipeline import FlowMatchingPipeline
except ImportError:  # Supports `python infer.py` inside flow_matching.
    from flow_matching.model import FlowMatchingUNet
    from flow_matching.pipeline import FlowMatchingPipeline

from utils.checkpoint import read_checkpoint
from utils.inference import save_generated_grid


def main():
    checkpoint_path = Path(__file__).with_name("checkpoints") / "flow_matching_latest.pt"
    output_path = Path(__file__).with_name("flow_matching_samples.png")
    batch_size = 16
    steps = 100

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = read_checkpoint(checkpoint_path, device)
    model_config = checkpoint.get("extra", {}).get("model_config")
    if model_config is None:
        raise KeyError("Checkpoint has no model_config. Train again with the current trainer.")

    model = FlowMatchingUNet(**model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    train_config = checkpoint["config"]
    pipeline = FlowMatchingPipeline(time_scale=train_config["time_scale"])
    with torch.no_grad():
        samples = pipeline.sample(
            model,
            batch_size=batch_size,
            image_shape=(model_config["in_channels"], train_config["image_size"], train_config["image_size"]),
            steps=steps,
            device=device,
        )

    save_generated_grid(samples.cpu(), output_path, nrow=4)
    print(f"Loaded: {checkpoint_path}")
    print(f"Saved samples: {output_path}")
    print(f"Euler steps: {steps}")


if __name__ == "__main__":
    main()
