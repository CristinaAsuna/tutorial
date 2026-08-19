"""VQGAN reconstruction inference using reusable utils modules."""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .vqgan import VQGAN
except ImportError:  # Supports `python infer.py` inside the vqgan directory.
    from vqgan import VQGAN

from utils.checkpoint import read_checkpoint
from utils.data import FlatImageDataset, build_image_transform
from utils.inference import save_reconstruction_grid


def main():
    checkpoint_path = Path(__file__).with_name("checkpoints") / "vqgan_latest.pt"
    output_path = Path(__file__).with_name("vqgan_reconstruction.png")
    batch_size = 8
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = read_checkpoint(checkpoint_path, device)
    train_config = checkpoint["config"]
    model_config = checkpoint.get("extra", {}).get("model_config")
    if model_config is None:
        # Compatibility with checkpoints created by the older all-in-one script.
        names = ("factor", "latent_dim", "code_dim", "num_codes", "disc_base_channels")
        model_config = {name: train_config[name] for name in names}
        model_config.update(in_channels=3, output_activation="tanh")

    model = VQGAN(**model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset = FlatImageDataset(
        train_config["data_root"],
        build_image_transform(train_config["image_size"], value_range="tanh"),
    )
    images = next(iter(DataLoader(dataset, batch_size=batch_size, shuffle=True))).to(device)
    with torch.no_grad():
        reconstructions, indices, _, metrics = model(images)

    save_reconstruction_grid(images.cpu(), reconstructions.cpu(), output_path, sample_count=batch_size)
    print(f"Loaded: {checkpoint_path}")
    print(f"Saved: {output_path}")
    print(f"Input/reconstruction: {tuple(images.shape)}")
    print(f"Token grid: {tuple(indices.shape)}")
    print(f"Perplexity: {metrics['perplexity'].item():.2f}")
    print(f"Active codes: {int(metrics['active_codes'].item())}/{model_config['num_codes']}")


if __name__ == "__main__":
    main()
