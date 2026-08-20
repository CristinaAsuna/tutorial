import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.data import FlatImageDataset, build_image_transform
from utils.inference import save_generated_grid, save_reconstruction_grid
from utils.losses import VQGANLoss


@dataclass
class VQGANTrainConfig:
    """数据、优化器与损失配置；模型结构参数不放在这里。"""

    data_root: str
    output_dir: str
    image_size: int = 64
    batch_size: int = 16
    epochs: int = 10
    num_workers: int = 4
    lr_generator: float = 2e-4
    lr_discriminator: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.9
    reconstruction_weight: float = 1.0
    perceptual_weight: float = 1.0
    perceptual_net: str = "vgg"
    discriminator_weight: float = 1.0
    discriminator_start: int = 1_000
    checkpoint_every: int = 1
    sample_count: int = 8
    seed: int = 42
    resume_checkpoint: str = ""


def set_requires_grad(module, requires_grad):
    for parameter in module.parameters():
        parameter.requires_grad_(requires_grad)


def find_last_conv_weight(module):
    """返回模块中最后一个 Conv2d 的权重，供 VQGAN adaptive GAN weight 使用。"""
    for layer in reversed(list(module.modules())):
        if isinstance(layer, nn.Conv2d):
            return layer.weight
    raise RuntimeError("No Conv2d layer found for adaptive GAN weighting.")


class VQGANTrainer:
    """可复用的 VQGAN 训练器。

    模型只需提供 ``generator``、``discriminator``、``discriminate``，并且
    ``forward(images)`` 返回 ``reconstruction, indices, vq_loss, metrics``。
    """

    def __init__(self, model, config: VQGANTrainConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = VQGANLoss(
            reconstruction_weight=config.reconstruction_weight,
            perceptual_weight=config.perceptual_weight,
            perceptual_net=config.perceptual_net,
            discriminator_weight=config.discriminator_weight,
            discriminator_start=config.discriminator_start,
        ).to(self.device)
        self.optimizers = {
            "generator": AdamW(
                self.model.generator.parameters(),
                lr=config.lr_generator,
                betas=(config.beta1, config.beta2),
            ),
            "discriminator": AdamW(
                self.model.discriminator.parameters(),
                lr=config.lr_discriminator,
                betas=(config.beta1, config.beta2),
            ),
        }
        self.global_step = 0
        self.start_epoch = 1
        self._restore_if_requested()

    def build_dataloader(self):
        transform = build_image_transform(self.config.image_size, value_range="tanh")
        dataset = FlatImageDataset(self.config.data_root, transform)
        workers = min(self.config.num_workers, os.cpu_count() or 1)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=workers > 0,
        )
        return dataset, loader

    def _restore_if_requested(self):
        if not self.config.resume_checkpoint:
            return
        checkpoint = load_checkpoint(
            self.config.resume_checkpoint,
            self.model,
            self.device,
            optimizers=self.optimizers,
        )
        self.global_step = checkpoint["global_step"]
        self.start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from {self.config.resume_checkpoint}: epoch {self.start_epoch}, step {self.global_step}")

    def _save(self, epoch):
        state_args = dict(
            model=self.model,
            epoch=epoch,
            global_step=self.global_step,
            optimizers=self.optimizers,
            config=self.config,
            extra={"model_config": self.model.get_config()} if hasattr(self.model, "get_config") else None,
        )
        save_checkpoint(self.output_dir / "vqgan_latest.pt", **state_args)
        if epoch % self.config.checkpoint_every == 0:
            save_checkpoint(self.output_dir / f"vqgan_epoch_{epoch:03d}.pt", **state_args)

    def fit(self):
        torch.manual_seed(self.config.seed)
        dataset, loader = self.build_dataloader()
        print(f"Device: {self.device} | images: {len(dataset):,} | output: {self.output_dir.resolve()}")

        for epoch in range(self.start_epoch, self.config.epochs + 1):
            self.model.train()
            running = defaultdict(float)
            progress = tqdm(loader, desc=f"Epoch {epoch}/{self.config.epochs}", unit="batch")
            last_images = last_reconstructions = None

            for images in progress:
                images = images.to(self.device, non_blocking=True)
                generator_optimizer = self.optimizers["generator"]
                discriminator_optimizer = self.optimizers["discriminator"]

                # Generator / tokenizer update: D is frozen but D(x_hat) still
                # backpropagates into the reconstruction tensor.
                set_requires_grad(self.model.discriminator, False)
                generator_optimizer.zero_grad(set_to_none=True)
                reconstructions, _, vq_loss, quant_metrics = self.model(images)
                gan_active = self.criterion.discriminator_factor(self.global_step) > 0.0
                fake_logits_g = self.model.discriminate(reconstructions) if gan_active else None
                generator_loss, generator_metrics = self.criterion.generator_loss(
                    target=images,
                    reconstruction=reconstructions,
                    vq_loss=vq_loss,
                    fake_logits=fake_logits_g,
                    global_step=self.global_step,
                    last_layer=find_last_conv_weight(self.model.generator.decoder),
                )
                generator_loss.backward()
                generator_optimizer.step()

                # Discriminator update: detach prevents a second generator update.
                discriminator_loss = images.new_zeros(())
                discriminator_metrics = {}
                if gan_active:
                    set_requires_grad(self.model.discriminator, True)
                    discriminator_optimizer.zero_grad(set_to_none=True)
                    real_logits = self.model.discriminate(images)
                    fake_logits_d = self.model.discriminate(reconstructions.detach())
                    discriminator_loss, discriminator_metrics = self.criterion.discriminator_loss(
                        real_logits, fake_logits_d, self.global_step
                    )
                    discriminator_loss.backward()
                    discriminator_optimizer.step()

                self.global_step += 1
                last_images, last_reconstructions = images.detach(), reconstructions.detach()
                metrics = {**generator_metrics, **quant_metrics, **discriminator_metrics}
                for name, value in metrics.items():
                    running[name] += float(value)
                progress.set_postfix(
                    g=f"{float(generator_loss.detach()):.4f}",
                    d=f"{float(discriminator_loss.detach()):.4f}",
                    pixel=f"{float(generator_metrics['pixel']):.4f}",
                    lpips=f"{float(generator_metrics['perceptual']):.4f}",
                    ppl=f"{float(quant_metrics['perplexity']):.1f}",
                )

            steps = len(loader)
            print(
                f"Epoch {epoch}/{self.config.epochs} | G: {running['generator_total'] / steps:.5f} | "
                f"D: {running['discriminator_total'] / steps:.5f} | "
                f"pixel: {running['pixel'] / steps:.5f} | "
                f"LPIPS: {running['perceptual'] / steps:.5f} | "
                f"perplexity: {running['perplexity'] / steps:.1f}"
            )
            save_reconstruction_grid(
                last_images.cpu(),
                last_reconstructions.cpu(),
                self.output_dir / f"reconstruction_epoch_{epoch:03d}.png",
                self.config.sample_count,
            )
            self._save(epoch)
        return self.model


@dataclass
class FlowMatchingTrainConfig:
    """通用图像 Flow Matching 训练配置。"""

    data_root: str
    output_dir: str
    image_size: int = 64
    batch_size: int = 16
    epochs: int = 10
    num_workers: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    time_scale: float = 1000.0
    sample_steps: int = 50
    sample_count: int = 16
    checkpoint_every: int = 1
    seed: int = 42
    resume_checkpoint: str = ""


class FlowMatchingTrainer:
    """训练任意满足 ``model(x_t, model_time) -> velocity`` 的网络。"""

    def __init__(self, model, pipeline, config: FlowMatchingTrainConfig):
        self.model = model
        self.pipeline = pipeline
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.global_step = 0
        self.start_epoch = 1
        self._restore_if_requested()

    def _restore_if_requested(self):
        if not self.config.resume_checkpoint:
            return
        checkpoint = load_checkpoint(
            self.config.resume_checkpoint,
            self.model,
            self.device,
            optimizers={"model": self.optimizer},
        )
        self.global_step = checkpoint["global_step"]
        self.start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from {self.config.resume_checkpoint}: epoch {self.start_epoch}, step {self.global_step}")

    def _build_dataloader(self):
        dataset = FlatImageDataset(
            self.config.data_root,
            build_image_transform(self.config.image_size, value_range="tanh"),
        )
        workers = min(self.config.num_workers, os.cpu_count() or 1)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=workers > 0,
        )
        return dataset, loader

    def _save(self, epoch):
        extra = {"model_config": self.model.get_config()} if hasattr(self.model, "get_config") else None
        args = dict(
            model=self.model,
            epoch=epoch,
            global_step=self.global_step,
            optimizers={"model": self.optimizer},
            config=self.config,
            extra=extra,
        )
        save_checkpoint(self.output_dir / "flow_matching_latest.pt", **args)
        if epoch % self.config.checkpoint_every == 0:
            save_checkpoint(self.output_dir / f"flow_matching_epoch_{epoch:03d}.pt", **args)

    def fit(self, dataloader=None, dataset_size=None):
        torch.manual_seed(self.config.seed)
        if dataloader is None:
            dataset, loader = self._build_dataloader()
            dataset_size = len(dataset)
        else:
            loader = dataloader
            dataset_size = len(loader.dataset) if dataset_size is None else dataset_size
        print(f"Device: {self.device} | images: {dataset_size:,} | output: {self.output_dir.resolve()}")

        for epoch in range(self.start_epoch, self.config.epochs + 1):
            self.model.train()
            total_loss = 0.0
            progress = tqdm(loader, desc=f"Epoch {epoch}/{self.config.epochs}", unit="batch")

            for images in progress:
                images = images.to(self.device, non_blocking=True)
                self.optimizer.zero_grad(set_to_none=True)
                loss, metrics = self.pipeline.training_loss(self.model, images)
                loss.backward()
                self.optimizer.step()

                self.global_step += 1
                total_loss += float(loss.detach())
                progress.set_postfix(loss=f"{float(loss.detach()):.5f}", t=f"{float(metrics['t']):.3f}")

            average_loss = total_loss / len(loader)
            print(f"Epoch {epoch}/{self.config.epochs} | velocity MSE: {average_loss:.6f}")

            self.model.eval()
            with torch.no_grad():
                samples = self.pipeline.sample(
                    self.model,
                    batch_size=self.config.sample_count,
                    image_shape=images.shape[1:],
                    steps=self.config.sample_steps,
                    device=self.device,
                )
            save_generated_grid(samples.cpu(), self.output_dir / f"samples_epoch_{epoch:03d}.png", nrow=4)
            self._save(epoch)

        return self.model


@dataclass
class ScoreSDETrainConfig:
    """连续时间 VP score-SDE 训练配置。"""

    output_dir: str
    epochs: int = 20
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    time_scale: float = 1000.0
    beta_min: float = 0.1
    beta_max: float = 20.0
    sde_eps: float = 1e-5
    sample_steps: int = 200
    sample_count: int = 16
    sample_every: int = 5
    seed: int = 42
    resume_checkpoint: str = ""


class ScoreSDETrainer:
    """用于 ``pipeline.training_loss`` 与 reverse-SDE sampler 的通用训练器。"""

    def __init__(self, model, pipeline, config: ScoreSDETrainConfig):
        self.model = model
        self.pipeline = pipeline
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.global_step = 0
        self.start_epoch = 1
        self._restore_if_requested()

    def _restore_if_requested(self):
        if not self.config.resume_checkpoint:
            return
        checkpoint = load_checkpoint(
            self.config.resume_checkpoint,
            self.model,
            self.device,
            optimizers={"model": self.optimizer},
        )
        self.global_step = checkpoint["global_step"]
        self.start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from {self.config.resume_checkpoint}: epoch {self.start_epoch}, step {self.global_step}")

    def _save_latest(self, epoch):
        extra = {"model_config": self.model.get_config()} if hasattr(self.model, "get_config") else None
        save_checkpoint(
            self.output_dir / "score_sde_latest.pt",
            model=self.model,
            epoch=epoch,
            global_step=self.global_step,
            optimizers={"model": self.optimizer},
            config=self.config,
            extra=extra,
        )

    def fit(self, dataloader):
        torch.manual_seed(self.config.seed)
        print(
            f"Device: {self.device} | images: {len(dataloader.dataset):,} | "
            f"output: {self.output_dir.resolve()}"
        )

        for epoch in range(self.start_epoch, self.config.epochs + 1):
            self.model.train()
            total_loss = 0.0
            progress = tqdm(dataloader, desc=f"Epoch {epoch}/{self.config.epochs}", unit="batch")
            image_shape = None

            for images in progress:
                images = images.to(self.device, non_blocking=True)
                image_shape = images.shape[1:]
                self.optimizer.zero_grad(set_to_none=True)
                loss, metrics = self.pipeline.training_loss(self.model, images)
                loss.backward()
                self.optimizer.step()

                self.global_step += 1
                total_loss += float(loss.detach())
                progress.set_postfix(
                    loss=f"{float(loss.detach()):.5f}",
                    t=f"{float(metrics['t']):.3f}",
                    std=f"{float(metrics['std']):.3f}",
                )

            print(f"Epoch {epoch}/{self.config.epochs} | weighted score loss: {total_loss / len(dataloader):.6f}")
            self._save_latest(epoch)  # overwritten each epoch; no epoch-by-epoch weight archive

            if epoch % self.config.sample_every == 0 or epoch == self.config.epochs:
                self.model.eval()
                samples = self.pipeline.sample_euler_maruyama(
                    self.model,
                    batch_size=self.config.sample_count,
                    image_shape=image_shape,
                    steps=self.config.sample_steps,
                    device=self.device,
                )
                save_generated_grid(
                    samples.cpu(),
                    self.output_dir / f"samples_epoch_{epoch:03d}.png",
                    nrow=4,
                )

        return self.model


@dataclass
class DDPMTrainConfig:
    """离散 DDPM 训练配置；每 epoch 只覆盖 latest checkpoint。"""

    output_dir: str
    epochs: int = 20
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    sample_count: int = 16
    sample_every: int = 5
    seed: int = 42
    resume_checkpoint: str = ""


class DDPMTrainer:
    """调用 DDPMPipeline.training_loss / sample 的通用训练器。"""

    def __init__(self, model, pipeline, config: DDPMTrainConfig):
        self.model = model
        self.pipeline = pipeline
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.optimizer = AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.global_step = 0
        self.start_epoch = 1

    def _save_latest(self, epoch):
        extra = {"model_config": self.model.get_config()} if hasattr(self.model, "get_config") else None
        save_checkpoint(
            self.output_dir / "ddpm_latest.pt",
            model=self.model,
            epoch=epoch,
            global_step=self.global_step,
            optimizers={"model": self.optimizer},
            config=self.config,
            extra=extra,
        )

    def fit(self, dataloader):
        torch.manual_seed(self.config.seed)
        print(f"Device: {self.device} | images: {len(dataloader.dataset):,} | output: {self.output_dir.resolve()}")

        for epoch in range(self.start_epoch, self.config.epochs + 1):
            self.model.train()
            total_loss = 0.0
            progress = tqdm(dataloader, desc=f"Epoch {epoch}/{self.config.epochs}", unit="batch")
            image_shape = None

            for images in progress:
                images = images.to(self.device, non_blocking=True)
                image_shape = images.shape[1:]
                self.optimizer.zero_grad(set_to_none=True)
                loss, metrics = self.pipeline.training_loss(self.model, images)
                loss.backward()
                self.optimizer.step()

                self.global_step += 1
                total_loss += float(loss.detach())
                progress.set_postfix(loss=f"{float(loss.detach()):.5f}", t=f"{float(metrics['t']):.1f}")

            print(f"Epoch {epoch}/{self.config.epochs} | epsilon MSE: {total_loss / len(dataloader):.6f}")
            self._save_latest(epoch)

            if epoch % self.config.sample_every == 0 or epoch == self.config.epochs:
                self.model.eval()
                samples = self.pipeline.sample(
                    self.model,
                    batch_size=self.config.sample_count,
                    image_shape=image_shape,
                    device=self.device,
                )
                save_generated_grid(samples.cpu(), self.output_dir / f"samples_epoch_{epoch:03d}.png", nrow=4)

        return self.model
