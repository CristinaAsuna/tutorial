import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from autoencoder import AE, AEencoder, AEdecoder


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 必须与训练时一致
    factor = 4
    latent = 64

    model = AE(
        encoder=AEencoder(inc=1, factor=factor, latent=latent),
        decoder=AEdecoder(inc=1, factor=factor, latent=latent),
    ).to(device)

    # 加载权重
    state_dict = torch.load("./mnist_ae.pth", map_location=device)
    model.load_state_dict(state_dict)

    # 推理模式：关闭 Dropout，固定 BatchNorm 等行为
    model.eval()

    test_dataset = datasets.MNIST(
        root=r"D:\Datasets",
        train=False,
        transform=transforms.ToTensor(),
        download=False,
    )
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

    x, labels = next(iter(test_loader))
    x = x.to(device)

    with torch.no_grad():
        reconstructed = model(x)

    # 保存原图与重建图对比
    x = x.cpu()
    reconstructed = reconstructed.cpu().clamp(0, 1)

    fig, axes = plt.subplots(2, 8, figsize=(16, 4))

    for i in range(8):
        axes[0, i].imshow(x[i, 0], cmap="gray")
        axes[0, i].set_title(f"Original: {labels[i]}")
        axes[0, i].axis("off")

        axes[1, i].imshow(reconstructed[i, 0], cmap="gray")
        axes[1, i].set_title("Reconstructed")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig("ae_reconstruction.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()