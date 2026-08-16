import torch
import sys
from pathlib import Path
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

# 支持在本目录中直接运行 `python vae.py`。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.attention import SpatialMHSA
from utils.basic_function import Residualblock


class VAEencoder(nn.Module):
    def __init__(self,inc=1,factor=8,latent=64,double_z=False):
        super().__init__()
        assert factor > 0 and factor & (factor - 1) == 0, "factor 必须是 2 的幂，例如 2、4、8"
        n_down = factor.bit_length() - 1
        out_latent=latent*2 if double_z else latent
        layers = [
            nn.Conv2d(inc, 128, kernel_size=3, padding=1),
            Residualblock(128, 128),
            Residualblock(128, 128),
        ]

        # factor=4 时循环 2 次；factor=8 时循环 3 次。
        # down_channels 记录每次下采样前的通道数，decoder 用它对称地恢复通道。
        self.down_channels = []
        channels = 128
        for i in range(n_down):
            self.down_channels.append(channels)
            layers.append(nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1))
            if i < n_down - 1:
                next_channels = min(channels * 2, 512)
                layers.extend([
                    Residualblock(channels, next_channels),
                    Residualblock(next_channels, next_channels),
                ])
                channels = next_channels

        layers.extend([
            Residualblock(channels, channels),
            Residualblock(channels, channels),
            Residualblock(channels, channels),
            SpatialMHSA(channels, nheads=2),
            Residualblock(channels, channels),
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_latent, kernel_size=3, padding=1),
        ])
        self.encode = nn.ModuleList(layers)
    def forward(self,x):
        for layer in self.encode:
            x=layer(x)
        return x
    
class VAEdecoder(nn.Module):
    def __init__(self,inc,factor=8,latent=64):
        super().__init__()
        assert factor > 0 and factor & (factor - 1) == 0, "factor 必须是 2 的幂，例如 2、4、8"
        n_down = factor.bit_length() - 1

        down_channels = []
        channels = 128
        for i in range(n_down):
            down_channels.append(channels)
            if i < n_down - 1:
                channels = min(channels * 2, 512)

        layers = [
            nn.Conv2d(latent, latent, kernel_size=1),
            nn.Conv2d(latent, channels, kernel_size=3, padding=1),
            Residualblock(channels, channels),
            SpatialMHSA(channels, nheads=2),
            Residualblock(channels, channels),
            Residualblock(channels, channels),
            Residualblock(channels, channels),
            Residualblock(channels, channels),
        ]

        # 按 encoder 的相反顺序上采样，factor=4 时恰好上采样两次。
        for target_channels in reversed(down_channels):
            layers.extend([
                nn.Upsample(scale_factor=2),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            ])
            if channels != target_channels:
                layers.append(Residualblock(channels, target_channels))
                channels = target_channels
            layers.extend([
                Residualblock(channels, channels),
                Residualblock(channels, channels),
            ])

        layers.extend([
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, inc, kernel_size=3, padding=1),
            nn.Sigmoid(),
        ])
        self.decode = nn.ModuleList(layers)
    def forward(self,x):
        for layer in self.decode:
            x=layer(x)
        return x

class KLVAE(nn.Module):
    
    def __init__(self,encoder,decoder,latent=64):
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder

        #initial conv for encode,decode
        self.quant_conv=nn.Conv2d(latent*2,latent*2,kernel_size=1)
        self.post_quant_conv=nn.Conv2d(latent,latent,kernel_size=1)

    def encode(self,x):
        
        moments=self.quant_conv(self.encoder(x))
        mu,logvar=moments.chunk(2,dim=1)
        # prvent exp explore
        logvar=logvar.clamp(-30,20)
        return mu,logvar
    
    def reparameterize(self,mu,logvar):
        #z=mu+std*noise
        std=torch.exp(0.5*logvar)
        noise=torch.randn_like(std)
        return mu+noise*std
    
    def decode(self,x):
        z=self.post_quant_conv(x)
        return self.decoder(z)
    
    def forward(self,x,sample_posterior=None):
        if sample_posterior is None:
            sample_posterior=self.training
        
        mu,logvar=self.encode(x)

        #training random sample
        #infer,mu
        z=self.reparameterize(mu,logvar) if sample_posterior else mu

        reconstructed = self.decode(z)

        return reconstructed,mu,logvar


class VAELoss(nn.Module):
    """KL-VAE 的训练目标：重建损失 + KL 损失 + 可选感知损失。"""

    def __init__(
        self,
        recon_weight=1.0,
        kl_weight=1e-6,
        perceptual_weight=0.0,
        perceptual_loss=None,
        recon_type="mse",
    ):
        super().__init__()
        if recon_type not in {"mse", "l1"}:
            raise ValueError("recon_type 必须是 'mse' 或 'l1'")
        if perceptual_weight > 0 and perceptual_loss is None:
            raise ValueError("启用 perceptual_weight 时必须传入 perceptual_loss")

        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.perceptual_weight = perceptual_weight
        self.perceptual_loss = perceptual_loss
        self.recon_type = recon_type

    def forward(self, reconstruction, target, mu, logvar):
        if self.recon_type == "mse":
            recon_loss = F.mse_loss(reconstruction, target)
        else:
            recon_loss = F.l1_loss(reconstruction, target)

        # 先对单张图片的所有 latent 维度求和，再对 batch 求均值。
        kl_loss = 0.5 * (
            mu.pow(2) + logvar.exp() - 1.0 - logvar
        ).sum(dim=(1, 2, 3)).mean()

        perceptual = reconstruction.new_zeros(())
        if self.perceptual_loss is not None:
            perceptual = self.perceptual_loss(reconstruction, target).mean()

        total_loss = (
            self.recon_weight * recon_loss
            + self.kl_weight * kl_loss
            + self.perceptual_weight * perceptual
        )

        return total_loss, {
            "total": total_loss.detach(),
            "recon": recon_loss.detach(),
            "kl": kl_loss.detach(),
            "perceptual": perceptual.detach(),
        }


def train_klvae(model, dataloader, epochs=20, lr=3e-4, kl_weight=1e-6, device="cpu"):
    """训练 KL-VAE；仅使用 MSE reconstruction loss 与 KL loss。"""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    loss_fn = VAELoss(recon_type="mse", kl_weight=kl_weight)

    for epoch in range(epochs):
        model.train()
        totals = {"total": 0.0, "recon": 0.0, "kl": 0.0}
        total_samples = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")
        for batch in progress_bar:
            # torchvision 数据集通常返回 (image, label)，KL-VAE 不使用 label。
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            reconstruction, mu, logvar = model(x, sample_posterior=True)
            loss, loss_dict = loss_fn(reconstruction, x, mu, logvar)

            loss.backward()
            optimizer.step()

            batch_size = x.size(0)
            total_samples += batch_size
            for name in totals:
                totals[name] += loss_dict[name].item() * batch_size

            progress_bar.set_postfix(
                total=f"{loss_dict['total'].item():.5f}",
                recon=f"{loss_dict['recon'].item():.5f}",
                kl=f"{loss_dict['kl'].item():.2f}",
            )

        averages = {name: value / total_samples for name, value in totals.items()}
        print(
            f"Epoch [{epoch + 1}/{epochs}] | "
            f"total={averages['total']:.6f} | "
            f"recon={averages['recon']:.6f} | kl={averages['kl']:.3f}"
        )

    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # MNIST 图片是 [0, 1] 的 (B, 1, 28, 28)，factor=4 可精确恢复尺寸。
    dataset = datasets.MNIST(
        root=r"D:\Datasets",
        train=True,
        transform=transforms.ToTensor(),
        download=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    factor = 4
    latent = 8
    model = KLVAE(
        encoder=VAEencoder(inc=1, factor=factor, latent=latent, double_z=True),
        decoder=VAEdecoder(inc=1, factor=factor, latent=latent),
        latent=latent,
    )

    model = train_klvae(
        model,
        dataloader,
        epochs=2,
        lr=3e-4,
        kl_weight=1e-6,
        device=device,
    )

    save_path = "mnist_klvae_f4_z8.pth"
    torch.save(model.state_dict(), save_path)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"Saved: {save_path}")
    print(f"Model parameters: {total:,}")


if __name__ == "__main__":
    main()


        
    
