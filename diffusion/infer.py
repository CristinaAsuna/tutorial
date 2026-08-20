"""DDPM inference entry point.

This becomes runnable after you implement DDPMPipeline.sample.
"""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .model import DDPMUNet
    from .pipeline import DDPMPipeline
except ImportError:
    from diffusion.model import DDPMUNet
    from diffusion.pipeline import DDPMPipeline

from utils.checkpoint import read_checkpoint
from utils.inference import save_generated_grid


def main():
    checkpoint_path = Path(__file__).with_name("checkpoints") / "ddpm_latest.pt"
    output_path = Path(__file__).with_name("ddpm_samples.png")
    batch_size = 16

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = read_checkpoint(checkpoint_path, device)
    model_config = checkpoint.get("extra", {}).get("model_config")
    if model_config is None:
        raise KeyError("Checkpoint has no model_config. Train using the current DDPM trainer first.")

    model = DDPMUNet(**model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Keep these schedule values identical to train.py.
    pipeline = DDPMPipeline(num_train_steps=1000, beta_start=1e-4, beta_end=2e-2)
    samples = pipeline.sample(model, batch_size, image_shape=(3, 32, 32), device=device)
    save_generated_grid(samples.cpu(), output_path, nrow=4)
    print(f"Loaded: {checkpoint_path}")
    print(f"Saved samples: {output_path}")


if __name__ == "__main__":
    main()
