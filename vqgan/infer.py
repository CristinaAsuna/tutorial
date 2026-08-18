import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vqgan.vqgan import FlatImageDataset, VQGAN


def main():
    # Change this path if you want to evaluate a particular epoch checkpoint.
    checkpoint_path = Path(__file__).with_name("checkpoints") / "vqgan_latest.pt"
    output_path = Path(__file__).with_name("vqgan_reconstruction.png")
    batch_size = 8

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run `python vqgan.py` first, or change checkpoint_path in infer.py."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = VQGAN(
        in_channels=3,
        factor=config["factor"],
        latent_dim=config["latent_dim"],
        code_dim=config["code_dim"],
        num_codes=config["num_codes"],
        disc_base_channels=config["disc_base_channels"],
        output_activation="tanh",
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(config["image_size"]),
        transforms.CenterCrop(config["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = FlatImageDataset(config["data_root"], transform)
    images = next(iter(DataLoader(dataset, batch_size=batch_size, shuffle=True)))
    images = images.to(device)

    with torch.no_grad():
        reconstructions, indices, _, metrics = model(images)

    # First row: originals; second row: reconstructions.
    comparison = torch.cat([images, reconstructions], dim=0)
    save_image((comparison.cpu() + 1.0) * 0.5, output_path, nrow=images.shape[0])

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Saved reconstruction grid: {output_path}")
    print(f"Input / reconstruction shape: {tuple(images.shape)}")
    print(f"Token grid shape: {tuple(indices.shape)}")
    print(f"Perplexity: {metrics['perplexity'].item():.2f}")
    print(f"Active codes: {int(metrics['active_codes'].item())}/{config['num_codes']}")


if __name__ == "__main__":
    main()
