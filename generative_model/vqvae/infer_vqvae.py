from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from vqvae import VQVAE


def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 必须与训练时的配置一致。
    factor=4
    code_dim=32
    num_codes=256
    weights_path=Path(__file__).with_name("mnist_vqvae_f4_d32_k256.pth")

    if not weights_path.exists():
        raise FileNotFoundError(
            f"找不到权重: {weights_path}\n"
            "请先运行 vqvae.py 训练模型，或修改 weights_path。"
        )

    model=VQVAE(
        inc=1,
        factor=factor,
        latent=code_dim,
        code_dim=code_dim,
        num_codes=num_codes,
    ).to(device)
    state_dict=torch.load(weights_path,map_location=device,weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    test_dataset=datasets.MNIST(
        root=r"D:\Datasets",
        train=False,
        transform=transforms.ToTensor(),
        download=False,
    )
    test_loader=DataLoader(test_dataset,batch_size=8,shuffle=True)
    images,labels=next(iter(test_loader))
    images=images.to(device)

    with torch.no_grad():
        reconstructions,indices,_,metrics=model(images)

    images=images.cpu()
    reconstructions=reconstructions.cpu().clamp(0,1)
    indices=indices.cpu()

    figure,axes=plt.subplots(3,len(images),figsize=(16,6))
    for index in range(len(images)):
        axes[0,index].imshow(images[index,0],cmap="gray",vmin=0,vmax=1)
        axes[0,index].set_title(f"Input: {labels[index]}")
        axes[0,index].axis("off")

        axes[1,index].imshow(reconstructions[index,0],cmap="gray",vmin=0,vmax=1)
        axes[1,index].set_title("Reconstruction")
        axes[1,index].axis("off")

        # token ID 网格可视化；颜色只表示不同 token，不代表像素强度。
        axes[2,index].imshow(indices[index],cmap="tab20",vmin=0,vmax=num_codes-1)
        axes[2,index].set_title("Token IDs")
        axes[2,index].axis("off")

    figure.tight_layout()
    output_path=Path(__file__).with_name("vqvae_reconstruction_tokens.png")
    figure.savefig(output_path,dpi=150)

    print(f"Saved: {output_path}")
    print(f"Perplexity: {metrics['perplexity'].item():.2f}")
    print(f"Active codes in this batch: {int(metrics['active_codes'].item())}/{num_codes}")
    print(f"Token grid shape: {tuple(indices.shape)}")
    plt.show()


if __name__=="__main__":
    main()
