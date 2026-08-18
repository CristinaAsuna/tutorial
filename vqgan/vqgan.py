import os
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vqvae.vqvae import VQ_VAE


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator: one unnormalized logit per image patch."""

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
            # Hinge loss uses raw logits; do not add Sigmoid.
            nn.Conv2d(base_channels * 8, 1, 4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class VQGANLoss(nn.Module):
    """L1 reconstruction + VQ + Hinge GAN loss; no perceptual loss."""

    def __init__(
        self,
        reconstruction_weight=1.0,
        discriminator_weight=1.0,
        discriminator_start=0,
        adaptive_weight=True,
        max_adaptive_weight=1e4,
    ):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.discriminator_weight = discriminator_weight
        self.discriminator_start = discriminator_start
        self.adaptive_weight = adaptive_weight
        self.max_adaptive_weight = max_adaptive_weight

    @staticmethod
    def generator_hinge_loss(fake_logits):
        return -fake_logits.mean()

    @staticmethod
    def discriminator_hinge_loss(real_logits, fake_logits):
        return 0.5 * (F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean())

    def discriminator_factor(self, global_step):
        return self.discriminator_weight if global_step >= self.discriminator_start else 0.0

    def calculate_adaptive_weight(self, reconstruction_loss, gan_loss, last_layer):
        reconstruction_grad = torch.autograd.grad(
            reconstruction_loss, last_layer, retain_graph=True, allow_unused=True
        )[0]
        gan_grad = torch.autograd.grad(gan_loss, last_layer, retain_graph=True, allow_unused=True)[0]
        if reconstruction_grad is None or gan_grad is None:
            return reconstruction_loss.new_tensor(1.0)
        return (reconstruction_grad.norm() / (gan_grad.norm() + 1e-4)).clamp(
            0.0, self.max_adaptive_weight
        ).detach()

    def generator_loss(
        self, target, reconstruction, vq_loss, fake_logits=None, global_step=0, last_layer=None
    ):
        reconstruction_loss = F.l1_loss(reconstruction, target)
        disc_factor = self.discriminator_factor(global_step)
        gan_loss = reconstruction.new_zeros(())
        gan_weight = reconstruction.new_zeros(())
        if fake_logits is not None and disc_factor > 0.0:
            gan_loss = self.generator_hinge_loss(fake_logits)
            gan_weight = (
                self.calculate_adaptive_weight(reconstruction_loss, gan_loss, last_layer)
                if self.adaptive_weight and last_layer is not None
                else reconstruction.new_tensor(1.0)
            )

        total_loss = self.reconstruction_weight * reconstruction_loss + vq_loss
        total_loss = total_loss + disc_factor * gan_weight * gan_loss
        return total_loss, {
            "generator_total": total_loss.detach(),
            "reconstruction": reconstruction_loss.detach(),
            "vq": vq_loss.detach(),
            "generator_gan": gan_loss.detach(),
            "gan_weight": gan_weight.detach(),
            "disc_factor": reconstruction.new_tensor(disc_factor),
        }

    def discriminator_loss(self, real_logits, fake_logits, global_step=0):
        disc_factor = self.discriminator_factor(global_step)
        loss = (
            disc_factor * self.discriminator_hinge_loss(real_logits, fake_logits)
            if disc_factor > 0.0
            else real_logits.new_zeros(())
        )
        return loss, {
            "discriminator_total": loss.detach(),
            "real_logits": real_logits.detach().mean(),
            "fake_logits": fake_logits.detach().mean(),
            "disc_factor": real_logits.new_tensor(disc_factor),
        }


class VQGAN(nn.Module):
    """VQ-VAE tokenizer/generator paired with a PatchGAN discriminator."""

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
            raise ValueError("latent_dim must equal code_dim until pre/post-quant 1x1 convolutions are added.")
        self.generator = VQ_VAE(
            inc=in_channels, factor=factor, latent=latent_dim, double_z=False,
            code_dim=code_dim, num_codes=num_codes,
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

    def encode(self, x):
        z_e = self.generator.encoder(x)
        z_q, indices, vq_loss, quant_metrics = self.generator.quantizer(z_e)
        z_q_st = z_e + (z_q - z_e).detach()
        return z_q_st, indices, vq_loss, quant_metrics

    def decode(self, z_q):
        return self.generator.decoder(z_q)

    def decode_indices(self, indices):
        if indices.ndim != 3:
            raise ValueError(f"indices must have shape (B,H,W), got {tuple(indices.shape)}")
        batch_size, height, width = indices.shape
        quantizer = self.generator.quantizer
        z_q = quantizer.embedding(indices.reshape(-1))
        z_q = z_q.view(batch_size, height, width, quantizer.code_dim)
        return self.decode(z_q.permute(0, 3, 1, 2).contiguous())

    def discriminate(self, x):
        return self.discriminator(x)

    def forward(self, x):
        z_q_st, indices, vq_loss, quant_metrics = self.encode(x)
        return self.decode(z_q_st), indices, vq_loss, quant_metrics


class FlatImageDataset(Dataset):
    """Load pictures from a flat directory, such as CelebA-HQ."""

    extensions = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(self, root, transform):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {self.root}")
        self.paths = sorted(path for path in self.root.rglob("*") if path.suffix.lower() in self.extensions)
        if not self.paths:
            raise RuntimeError(f"No image files found in: {self.root}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


@dataclass
class TrainConfig:
    data_root: str = r"D:\Datasets\celebA\celeba_hq_256"
    output_dir: str = str(Path(__file__).with_name("checkpoints"))
    image_size: int = 64
    batch_size: int = 32
    epochs: int = 5
    num_workers: int = 8
    lr_generator: float = 2e-4
    lr_discriminator: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.9
    factor: int = 8
    latent_dim: int = 64
    code_dim: int = 64
    num_codes: int = 1024
    disc_base_channels: int = 64
    discriminator_start: int = 10_000
    checkpoint_every: int = 1
    sample_count: int = 8
    seed: int = 42
    resume_checkpoint: str = ""


def set_requires_grad(module, requires_grad):
    for parameter in module.parameters():
        parameter.requires_grad_(requires_grad)


def get_decoder_last_layer(model):
    """Find the final decoder Conv2d for adaptive GAN weighting."""
    for layer in reversed(model.generator.decoder.decode):
        if isinstance(layer, nn.Conv2d):
            return layer.weight
    raise RuntimeError("No Conv2d layer found in the decoder.")


def save_reconstruction_grid(images, reconstructions, path, sample_count):
    count = min(sample_count, images.shape[0])
    comparison = torch.cat([images[:count], reconstructions[:count]], dim=0)
    # The training range is [-1, 1]; PNG output needs [0, 1].
    save_image((comparison + 1.0) * 0.5, path, nrow=count)


def save_checkpoint(path, model, generator_optimizer, discriminator_optimizer, epoch, global_step, config):
    torch.save(
        {
            "model": model.state_dict(),
            "generator_optimizer": generator_optimizer.state_dict(),
            "discriminator_optimizer": discriminator_optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "config": asdict(config),
        },
        path,
    )


def train_vqgan(config=TrainConfig()):
    """Train VQGAN: generator/tokenizer first, then PatchGAN discriminator."""
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.CenterCrop(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = FlatImageDataset(config.data_root, transform)
    workers = min(config.num_workers, os.cpu_count() or 1)
    loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, num_workers=workers,
        pin_memory=device.type == "cuda", persistent_workers=workers > 0,
    )

    model = VQGAN(
        in_channels=3, factor=config.factor, latent_dim=config.latent_dim,
        code_dim=config.code_dim, num_codes=config.num_codes,
        disc_base_channels=config.disc_base_channels, output_activation="tanh",
    ).to(device)
    criterion = VQGANLoss(discriminator_start=config.discriminator_start)
    generator_optimizer = AdamW(model.generator.parameters(), lr=config.lr_generator, betas=(config.beta1, config.beta2))
    discriminator_optimizer = AdamW(model.discriminator.parameters(), lr=config.lr_discriminator, betas=(config.beta1, config.beta2))
    print(f"Device: {device} | images: {len(dataset):,} | output: {output_dir.resolve()}")
    global_step = 0
    start_epoch = 1
    if config.resume_checkpoint:
        resume_path = Path(config.resume_checkpoint)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        generator_optimizer.load_state_dict(checkpoint["generator_optimizer"])
        discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer"])
        global_step = checkpoint["global_step"]
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from {resume_path}: epoch {start_epoch}, global step {global_step}")

    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        running = defaultdict(float)
        progress = tqdm(loader, desc=f"Epoch {epoch}/{config.epochs}", unit="batch")
        last_images = last_reconstructions = None

        for images in progress:
            images = images.to(device, non_blocking=True)

            # Freeze D parameters, but retain the path D(x_hat) -> x_hat -> generator.
            set_requires_grad(model.discriminator, False)
            generator_optimizer.zero_grad(set_to_none=True)
            reconstructions, _, vq_loss, quant_metrics = model(images)
            gan_is_active = criterion.discriminator_factor(global_step) > 0.0
            fake_logits_g = model.discriminate(reconstructions) if gan_is_active else None
            generator_loss, generator_metrics = criterion.generator_loss(
                images, reconstructions, vq_loss, fake_logits_g,
                global_step, get_decoder_last_layer(model),
            )
            generator_loss.backward()
            generator_optimizer.step()

            # detach prevents this discriminator pass from updating the generator.
            discriminator_loss = images.new_zeros(())
            discriminator_metrics = {}
            if criterion.discriminator_factor(global_step) > 0.0:
                set_requires_grad(model.discriminator, True)
                discriminator_optimizer.zero_grad(set_to_none=True)
                real_logits = model.discriminate(images)
                fake_logits_d = model.discriminate(reconstructions.detach())
                discriminator_loss, discriminator_metrics = criterion.discriminator_loss(
                    real_logits, fake_logits_d, global_step
                )
                discriminator_loss.backward()
                discriminator_optimizer.step()

            global_step += 1
            last_images, last_reconstructions = images.detach(), reconstructions.detach()
            batch_metrics = {**generator_metrics, **quant_metrics, **discriminator_metrics}
            for name, value in batch_metrics.items():
                running[name] += float(value)
            progress.set_postfix(
                g=f"{float(generator_loss.detach()):.4f}",
                d=f"{float(discriminator_loss.detach()):.4f}",
                recon=f"{float(generator_metrics['reconstruction']):.4f}",
                ppl=f"{float(quant_metrics['perplexity']):.1f}",
            )

        steps = len(loader)
        print(
            f"Epoch {epoch}/{config.epochs} | G: {running['generator_total'] / steps:.5f} | "
            f"D: {running['discriminator_total'] / steps:.5f} | "
            f"recon: {running['reconstruction'] / steps:.5f} | "
            f"perplexity: {running['perplexity'] / steps:.1f}"
        )
        save_reconstruction_grid(
            last_images.cpu(), last_reconstructions.cpu(),
            output_dir / f"reconstruction_epoch_{epoch:03d}.png", config.sample_count,
        )
        save_checkpoint(
            output_dir / "vqgan_latest.pt", model, generator_optimizer,
            discriminator_optimizer, epoch, global_step, config,
        )
        if epoch % config.checkpoint_every == 0:
            save_checkpoint(
                output_dir / f"vqgan_epoch_{epoch:03d}.pt", model, generator_optimizer,
                discriminator_optimizer, epoch, global_step, config,
            )
    return model


def main():
    config = TrainConfig(
        epochs=10,
        image_size=64,          # 先用 64，速度明显更快
        batch_size=32,          # 显存不足就改为 16 / 8
        num_codes=256,
        latent_dim=32,
        code_dim=32,
        discriminator_start=1_000,  # 不要维持默认 10000
    )
    train_vqgan(config)


if __name__ == "__main__":
    main()
