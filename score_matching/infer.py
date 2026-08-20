"""Generate CIFAR-10-sized images using reverse VP-SDE sampling."""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .model import ScoreMatchingUNet
    from .pipeline import ScoreMatchingPipeline, VPSDE
except ImportError:  # Supports `python infer.py` inside score_matching.
    from score_matching.model import ScoreMatchingUNet
    from score_matching.pipeline import ScoreMatchingPipeline, VPSDE

from utils.checkpoint import read_checkpoint
from utils.inference import save_generated_grid


def main():
    checkpoint_path = Path(__file__).with_name("checkpoints") / "score_sde_latest.pt"
    output_path = Path(__file__).with_name("score_sde_samples.png")
    batch_size = 16
    steps = 200

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = read_checkpoint(checkpoint_path, device)
    model_config = checkpoint.get("extra", {}).get("model_config")
    if model_config is None:
        raise KeyError("Checkpoint has no model_config. Train with the current score-SDE trainer first.")

    model = ScoreMatchingUNet(**model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    config = checkpoint["config"]
    sde = VPSDE(config["beta_min"], config["beta_max"], config["sde_eps"])
    pipeline = ScoreMatchingPipeline(sde, time_scale=config["time_scale"])
    samples = pipeline.sample_euler_maruyama(
        model,
        batch_size=batch_size,
        image_shape=(model_config["in_channels"], 32, 32),
        steps=steps,
        device=device,
    )
    save_generated_grid(samples.cpu(), output_path, nrow=4)

    print(f"Loaded: {checkpoint_path}")
    print(f"Saved samples: {output_path}")
    print(f"Euler-Maruyama steps: {steps}")


if __name__ == "__main__":
    main()
