import torch
from torch import nn
from torch.nn import functional as F


class ReconstructionLoss(nn.Module):
    """可复用的像素重建损失。"""

    def __init__(self, kind="l1"):
        super().__init__()
        if kind not in {"l1", "mse"}:
            raise ValueError("kind must be 'l1' or 'mse'.")
        self.kind = kind

    def forward(self, reconstruction, target):
        return F.l1_loss(reconstruction, target) if self.kind == "l1" else F.mse_loss(reconstruction, target)


class KLDivergenceLoss(nn.Module):
    """KL(q(z|x) || N(0, I))，用于 KL-VAE。"""

    def forward(self, mu, logvar):
        return 0.5 * (mu.square() + logvar.exp() - 1.0 - logvar).sum(dim=(1, 2, 3)).mean()


class LPIPSLoss(nn.Module):
    """冻结的预训练 LPIPS 感知损失；输入必须为 RGB [-1,1] 张量。"""

    def __init__(self, network="vgg"):
        super().__init__()
        try:
            import lpips
        except ImportError as error:
            raise ImportError("Install LPIPS first: python -m pip install lpips") from error
        self.model = lpips.LPIPS(net=network).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def forward(self, reconstruction, target):
        return self.model(reconstruction, target).mean()


class VQGANLoss(nn.Module):
    """经典 VQGAN：pixel + LPIPS + VQ + adaptive Hinge GAN。"""

    def __init__(
        self,
        reconstruction_weight=1.0,
        perceptual_weight=1.0,
        perceptual_net="vgg",
        discriminator_weight=1.0,
        discriminator_start=0,
        adaptive_weight=True,
        max_adaptive_weight=1e4,
    ):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.perceptual_weight = perceptual_weight
        self.discriminator_weight = discriminator_weight
        self.discriminator_start = discriminator_start
        self.adaptive_weight = adaptive_weight
        self.max_adaptive_weight = max_adaptive_weight
        self.reconstruction_loss = ReconstructionLoss("l1")
        self.perceptual_loss = LPIPSLoss(perceptual_net) if perceptual_weight > 0.0 else None

    def discriminator_factor(self, global_step):
        return self.discriminator_weight if global_step >= self.discriminator_start else 0.0

    @staticmethod
    def generator_hinge_loss(fake_logits):
        return -fake_logits.mean()

    @staticmethod
    def discriminator_hinge_loss(real_logits, fake_logits):
        return 0.5 * (F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean())

    def calculate_adaptive_weight(self, image_loss, gan_loss, last_layer):
        image_grad = torch.autograd.grad(image_loss, last_layer, retain_graph=True, allow_unused=True)[0]
        gan_grad = torch.autograd.grad(gan_loss, last_layer, retain_graph=True, allow_unused=True)[0]
        if image_grad is None or gan_grad is None:
            return image_loss.new_tensor(1.0)
        return (image_grad.norm() / (gan_grad.norm() + 1e-4)).clamp(0.0, self.max_adaptive_weight).detach()

    def generator_loss(self, target, reconstruction, vq_loss, fake_logits=None, global_step=0, last_layer=None):
        pixel_loss = self.reconstruction_loss(reconstruction, target)
        perceptual_loss = reconstruction.new_zeros(())
        if self.perceptual_loss is not None:
            perceptual_loss = self.perceptual_loss(reconstruction, target)
        image_loss = self.reconstruction_weight * pixel_loss + self.perceptual_weight * perceptual_loss

        disc_factor = self.discriminator_factor(global_step)
        gan_loss = reconstruction.new_zeros(())
        gan_weight = reconstruction.new_zeros(())
        if fake_logits is not None and disc_factor > 0.0:
            gan_loss = self.generator_hinge_loss(fake_logits)
            gan_weight = (
                self.calculate_adaptive_weight(image_loss, gan_loss, last_layer)
                if self.adaptive_weight and last_layer is not None
                else reconstruction.new_tensor(1.0)
            )

        total_loss = image_loss + vq_loss + disc_factor * gan_weight * gan_loss
        return total_loss, {
            "generator_total": total_loss.detach(),
            "pixel": pixel_loss.detach(),
            "perceptual": perceptual_loss.detach(),
            "image": image_loss.detach(),
            "vq": vq_loss.detach(),
            "generator_gan": gan_loss.detach(),
            "gan_weight": gan_weight.detach(),
            "disc_factor": reconstruction.new_tensor(disc_factor),
        }

    def discriminator_loss(self, real_logits, fake_logits, global_step=0):
        disc_factor = self.discriminator_factor(global_step)
        loss = (
            disc_factor * self.discriminator_hinge_loss(real_logits, fake_logits)
            if disc_factor > 0.0 else real_logits.new_zeros(())
        )
        return loss, {
            "discriminator_total": loss.detach(),
            "real_logits": real_logits.detach().mean(),
            "fake_logits": fake_logits.detach().mean(),
            "disc_factor": real_logits.new_tensor(disc_factor),
        }
