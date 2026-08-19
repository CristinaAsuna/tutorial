"""VQGAN model architecture.

Training, losses, datasets, checkpoints, and visualization live in utils/.
"""

import sys
from pathlib import Path

from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vqvae.vqvae import VQ_VAE


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator that returns one raw logit per local patch."""

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
            nn.Conv2d(base_channels * 8, 1, 4, stride=1, padding=1),
        )

    def forward(self, images):
        return self.net(images)


class VQGAN(nn.Module):
    """VQ tokenizer/generator plus a PatchGAN discriminator.

    Its forward contract is shared by ``utils.trainers.VQGANTrainer``:
    ``reconstruction, indices, vq_loss, quant_metrics = model(images)``.
    """

    def __init__(
        self,
        in_channels=3,
        factor=8,
        latent_dim=32,
        code_dim=32,
        num_codes=256,
        disc_base_channels=64,
        output_activation="tanh",
    ):
        super().__init__()
        if latent_dim != code_dim:
            raise ValueError("latent_dim must equal code_dim until pre/post-quant 1x1 convolutions are added.")

        self.generator = VQ_VAE(
            inc=in_channels,
            factor=factor,
            latent=latent_dim,
            double_z=False,
            code_dim=code_dim,
            num_codes=num_codes,
        )
        if output_activation == "tanh":
            self.generator.decoder.decode[-1] = nn.Tanh()
        elif output_activation == "sigmoid":
            self.generator.decoder.decode[-1] = nn.Sigmoid()
        elif output_activation == "none":
            self.generator.decoder.decode[-1] = nn.Identity()
        else:
            raise ValueError("output_activation must be 'tanh', 'sigmoid', or 'none'.")

        self.discriminator = PatchDiscriminator(in_channels, disc_base_channels)
        self.model_config = {
            "in_channels": in_channels,
            "factor": factor,
            "latent_dim": latent_dim,
            "code_dim": code_dim,
            "num_codes": num_codes,
            "disc_base_channels": disc_base_channels,
            "output_activation": output_activation,
        }

    def get_config(self):
        """Architecture metadata used by generic checkpoint/inference utilities."""
        return dict(self.model_config)

    def encode(self, images):
        z_e = self.generator.encoder(images)
        z_q, indices, vq_loss, quant_metrics = self.generator.quantizer(z_e)
        z_q_st = z_e + (z_q - z_e).detach()
        return z_q_st, indices, vq_loss, quant_metrics

    def decode(self, latent):
        return self.generator.decoder(latent)

    def decode_indices(self, indices):
        """Decode discrete token IDs with shape (B, H_token, W_token)."""
        if indices.ndim != 3:
            raise ValueError(f"indices must have shape (B,H,W), got {tuple(indices.shape)}")
        batch_size, height, width = indices.shape
        quantizer = self.generator.quantizer
        z_q = quantizer.embedding(indices.reshape(-1))
        z_q = z_q.view(batch_size, height, width, quantizer.code_dim)
        return self.decode(z_q.permute(0, 3, 1, 2).contiguous())

    def discriminate(self, images):
        return self.discriminator(images)

    def forward(self, images):
        z_q_st, indices, vq_loss, quant_metrics = self.encode(images)
        reconstruction = self.decode(z_q_st)
        return reconstruction, indices, vq_loss, quant_metrics


def main():
    # Future projects only need to change model architecture above and this config.
    from utils.trainers import VQGANTrainConfig, VQGANTrainer

    model = VQGAN(
        in_channels=3,
        factor=8,
        latent_dim=32,
        code_dim=32,
        num_codes=256,
    )
    config = VQGANTrainConfig(
        data_root=r"D:\Datasets\celebA\celeba_hq_256",
        output_dir=str(Path(__file__).with_name("checkpoints")),
        image_size=64,
        batch_size=32,
        epochs=10,
        discriminator_start=1_000,
        perceptual_weight=1.0,
    )
    VQGANTrainer(model, config).fit()


if __name__ == "__main__":
    main()
