from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from vae import KLVAE, VAEdecoder, VAEencoder


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 必须与训练时的模型配置完全一致。
    factor = 4
    latent = 8
    weights_path = Path(__file__).with_name("mnist_klvae_f4_z8.pth")

    if not weights_path.exists():
        raise FileNotFoundError(
            f"找不到权重文件: {weights_path}\n"
            "请先运行 vae.py 完成训练，或修改 weights_path。"
        )

    model = KLVAE(
        encoder=VAEencoder(inc=1, factor=factor, latent=latent, double_z=True),
        decoder=VAEdecoder(inc=1, factor=factor, latent=latent),
        latent=latent,
    ).to(device)

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    test_dataset = datasets.MNIST(
        root=r"D:\Datasets",
        train=False,
        transform=transforms.ToTensor(),
        download=False,
    )
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

    images, labels = next(iter(test_loader))
    images = images.to(device)

    with torch.no_grad():
        # 推理用 posterior mean，而非随机采样，保证每次结果一致。
        reconstructions, mu, logvar = model(images, sample_posterior=False)

    images = images.cpu()
    reconstructions = reconstructions.cpu().clamp(0, 1)

    figure, axes = plt.subplots(2, len(images), figsize=(16, 4))
    for index in range(len(images)):
        axes[0, index].imshow(images[index, 0], cmap="gray", vmin=0, vmax=1)
        axes[0, index].set_title(f"Input: {labels[index]}")
        axes[0, index].axis("off")

        axes[1, index].imshow(reconstructions[index, 0], cmap="gray", vmin=0, vmax=1)
        axes[1, index].set_title("Reconstruction")
        axes[1, index].axis("off")

    figure.tight_layout()
    output_path = Path(__file__).with_name("klvae_reconstruction.png")
    figure.savefig(output_path, dpi=150)
    print(f"Saved reconstruction comparison: {output_path}")
    print(f"mu shape: {tuple(mu.shape)}, logvar shape: {tuple(logvar.shape)}")
    plt.show()


if __name__ == "__main__":
    main()
