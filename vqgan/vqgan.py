import sys
from pathlib import Path

import torch
from torch import nn


# 允许在 vqgan 目录中直接运行 `python vqgan.py`。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vqvae.vqvae import VQ_VAE


class PatchDiscriminator(nn.Module):
    """对图像局部 patch 输出真假 logits 的判别器。"""

    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 4, base_channels * 8, 4, stride=1, padding=1),
            nn.InstanceNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Hinge GAN loss 直接使用 logits，因此不接 Sigmoid。
            nn.Conv2d(base_channels * 8, 1, 4, stride=1, padding=1),
        )

    def forward(self, x):
        #(b,c,h,w)-->(b,1,h',w')
        return self.net(x)


class VQGAN(nn.Module):
    """VQ-VAE generator 与 PatchGAN discriminator 的模型容器。"""

    def __init__(
        self,
        in_channels=3,
        factor=8,
        latent_dim=64,
        code_dim=64,
        num_codes=1024,
        disc_base_channels=64,
        output_activation="tanh",
    ):
        super().__init__()

        if latent_dim != code_dim:
            raise ValueError(
                "当前 VQ_VAE 没有 pre/post quant 1×1 Conv，"
                "因此 latent_dim 必须等于 code_dim。"
            )

        self.generator = VQ_VAE(
            inc=in_channels,
            factor=factor,
            latent=latent_dim,
            double_z=False,
            code_dim=code_dim,
            num_codes=num_codes,
        )

        # CelebA 的推荐输入范围为 [-1, 1]，因此使用 Tanh 输出。
        if output_activation == "tanh":
            self.generator.decoder.decode[-1] = nn.Tanh()
        elif output_activation == "sigmoid":
            self.generator.decoder.decode[-1] = nn.Sigmoid()
        elif output_activation == "none":
            self.generator.decoder.decode[-1] = nn.Identity()
        else:
            raise ValueError("output_activation 必须为 tanh、sigmoid 或 none")

        self.discriminator = PatchDiscriminator(
            in_channels=in_channels,
            base_channels=disc_base_channels,
        )

    def encode(self, x):
        """图像编码为量化 latent、离散 token IDs 与 VQ loss。"""
        z_e = self.generator.encoder(x)
        z_q, indices, vq_loss, metrics = self.generator.quantizer(z_e)

        # Straight-Through Estimator：前向用 z_q，反向向 encoder 传梯度。
        z_q_st = z_e + (z_q - z_e).detach()
        return z_q_st, indices, vq_loss, metrics

    def decode(self, z_q):
        return self.generator.decoder(z_q)

    def decode_indices(self, indices):
        """将 Transformer/MaskGIT 生成的 token IDs 解码为图像。"""
        if indices.ndim != 3:
            raise ValueError(f"indices 应为 (B,H,W)，实际为 {tuple(indices.shape)}")

        batch_size, height, width = indices.shape
        quantizer = self.generator.quantizer
        z_q = quantizer.embedding(indices.reshape(-1))
        z_q = z_q.view(batch_size, height, width, quantizer.code_dim)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return self.decode(z_q)

    def discriminate(self, x):
        return self.discriminator(x)

    def forward(self, x):
        """返回重建图、token IDs、VQ loss 与 tokenizer 指标。"""
        z_q_st, indices, vq_loss, metrics = self.encode(x)
        reconstruction = self.decode(z_q_st)
        return reconstruction, indices, vq_loss, metrics


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VQGAN(
        in_channels=3,
        factor=8,
        latent_dim=64,
        code_dim=64,
        num_codes=1024,
    ).to(device)

    images = torch.randn(2, 3, 64, 64, device=device)
    with torch.no_grad():
        reconstructions, indices, vq_loss, metrics = model(images)
        patch_logits = model.discriminate(reconstructions)

    print(f"input: {tuple(images.shape)}")
    print(f"reconstruction: {tuple(reconstructions.shape)}")
    print(f"token IDs: {tuple(indices.shape)}")
    print(f"PatchGAN logits: {tuple(patch_logits.shape)}")
    print(f"VQ loss: {vq_loss.item():.6f}")
    print(f"perplexity: {metrics['perplexity'].item():.2f}")
