import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from clip import Clip
from ddpm import DDPMSampler
from decoder import VAE_Decoder
from diffusion import Diffusion
from encoder import VAE_Encoder
from pipeline import generate

try:
    import swanlab
except ImportError:  # pragma: no cover - optional dependency
    swanlab = None


def parse_args():
    parser = argparse.ArgumentParser(description="Train VAE or latent diffusion on a Hugging Face dataset.")
    parser.add_argument("--stage", choices=["vae", "diffusion"], default="diffusion")
    parser.add_argument("--dataset_name", default="reach-vb/pokemon-blip-captions")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--validation_split", type=float, default=0.05)
    parser.add_argument("--validation_split_name", default=None)
    parser.add_argument("--image_column", default=None)
    parser.add_argument("--caption_column", default=None)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--kl_weight", type=float, default=1e-6)
    parser.add_argument("--cfg_dropout_prob", type=float, default=0.1)
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--save_weights_only", action="store_true")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--validate_every", type=int, default=500)
    parser.add_argument("--sample_every", type=int, default=500)
    parser.add_argument("--num_val_batches", type=int, default=10)
    parser.add_argument("--num_sample_prompts", type=int, default=4)
    parser.add_argument("--sample_prompts", nargs="*", default=None)
    parser.add_argument("--resume_from", default=None)
    parser.add_argument("--pretrained_checkpoint", default=None)
    parser.add_argument("--pretrained_format", choices=["train", "standard"], default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--use_swanlab", action="store_true")
    parser.add_argument("--swanlab_project", default="ldm-sd")
    parser.add_argument("--train_clip", action="store_true")
    parser.add_argument("--train_encoder", action="store_true")
    parser.add_argument("--train_decoder", action="store_true")
    parser.add_argument("--tokenizer_name", default="openai/clip-vit-large-patch14")
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_column_name(column_names: List[str], explicit_name: Optional[str], candidates: List[str], kind: str) -> str:
    if explicit_name:
        if explicit_name not in column_names:
            raise ValueError(f"{kind} column '{explicit_name}' not found in dataset columns: {column_names}")
        return explicit_name

    for candidate in candidates:
        if candidate in column_names:
            return candidate

    raise ValueError(f"Could not infer {kind} column from dataset columns: {column_names}")


def preprocess_image(image: Image.Image, resolution: int) -> torch.Tensor:
    image = image.convert("RGB").resize((resolution, resolution))
    array = np.asarray(image, dtype=np.float32)
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    tensor = tensor / 127.5 - 1.0
    return tensor


class ImageTextDataset(Dataset):
    def __init__(self, dataset, image_column: str, caption_column: Optional[str], resolution: int):
        self.dataset = dataset
        self.image_column = image_column
        self.caption_column = caption_column
        self.resolution = resolution

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = preprocess_image(item[self.image_column], self.resolution)
        caption = ""
        if self.caption_column is not None:
            caption = str(item[self.caption_column])
        return {"pixel_values": image, "caption": caption}


def collate_examples(examples: List[Dict[str, torch.Tensor]]):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    captions = [example["caption"] for example in examples]
    return {"pixel_values": pixel_values, "captions": captions}


def get_time_embedding_batch(timesteps: torch.LongTensor, dim: int = 320) -> torch.FloatTensor:
    half_dim = dim // 2
    freqs = torch.pow(
        10000,
        -torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device) / half_dim,
    )
    x = timesteps.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)


def add_noise_for_training(latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.LongTensor, alphas_cumprod: torch.Tensor):
    alphas_cumprod = alphas_cumprod.to(device=latents.device, dtype=latents.dtype)
    sqrt_alpha_prod = alphas_cumprod[timesteps].sqrt().view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]).sqrt().view(-1, 1, 1, 1)
    return sqrt_alpha_prod * latents + sqrt_one_minus_alpha_prod * noise


def maybe_load_tokenizer(tokenizer_name: str):
    try:
        from transformers import CLIPTokenizer
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ImportError("transformers is required for diffusion training. Please install it first.") from exc

    return CLIPTokenizer.from_pretrained(tokenizer_name)


def maybe_load_pretrained_models(models: Dict[str, nn.Module], args, device: str):
    if not args.pretrained_checkpoint:
        return

    if args.pretrained_format == "train":
        state = torch.load(args.pretrained_checkpoint, map_location=device, weights_only=False)
        for name, model in models.items():
            if name in state:
                model.load_state_dict(state[name], strict=True)
        return

    from model_loader import preload_models_from_standard_weights

    pretrained_models = preload_models_from_standard_weights(args.pretrained_checkpoint, device)
    for name, model in models.items():
        if name in pretrained_models:
            model.load_state_dict(pretrained_models[name].state_dict(), strict=True)


def move_state_to_cpu(state):
    if torch.is_tensor(state):
        return state.detach().cpu()
    if isinstance(state, dict):
        return {key: move_state_to_cpu(value) for key, value in state.items()}
    if isinstance(state, list):
        return [move_state_to_cpu(value) for value in state]
    if isinstance(state, tuple):
        return tuple(move_state_to_cpu(value) for value in state)
    return state


def save_checkpoint(
    output_dir: Path,
    global_step: int,
    epoch: int,
    models: Dict[str, nn.Module],
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    args,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "global_step": global_step,
        "epoch": epoch,
        "args": vars(args),
    }
    if not args.save_weights_only:
        state["optimizer"] = move_state_to_cpu(optimizer.state_dict())
        state["scaler"] = move_state_to_cpu(scaler.state_dict()) if scaler is not None else None
    for name, model in models.items():
        state[name] = move_state_to_cpu(model.state_dict())

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    step_path = output_dir / f"checkpoint_step_{global_step}.pt"
    latest_path = output_dir / "latest.pt"
    torch.save(state, step_path)
    torch.save(state, latest_path)


def save_best_checkpoint(
    output_dir: Path,
    global_step: int,
    epoch: int,
    models: Dict[str, nn.Module],
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    args,
    metric_name: str,
    metric_value: float,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "global_step": global_step,
        "epoch": epoch,
        "args": vars(args),
        "best_metric_name": metric_name,
        "best_metric_value": metric_value,
    }
    if not args.save_weights_only:
        state["optimizer"] = move_state_to_cpu(optimizer.state_dict())
        state["scaler"] = move_state_to_cpu(scaler.state_dict()) if scaler is not None else None
    for name, model in models.items():
        state[name] = move_state_to_cpu(model.state_dict())

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    best_path = output_dir / "best.pt"
    torch.save(state, best_path)


def load_checkpoint(checkpoint_path: str, models: Dict[str, nn.Module], optimizer: torch.optim.Optimizer, scaler: Optional[GradScaler], device: str):
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    for name, model in models.items():
        if name in state:
            model.load_state_dict(state[name], strict=True)

    if "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])

    return state.get("epoch", 0), state.get("global_step", 0)


def build_models(device: str):
    models = {
        "encoder": VAE_Encoder().to(device),
        "decoder": VAE_Decoder().to(device),
        "clip": Clip().to(device),
        "diffusion": Diffusion().to(device),
    }
    return models


def set_trainable(models: Dict[str, nn.Module], args):
    if args.stage == "vae":
        trainable = {"encoder", "decoder"}
    else:
        trainable = {"diffusion"}
        if args.train_clip:
            trainable.add("clip")
        if args.train_encoder:
            trainable.add("encoder")
        if args.train_decoder:
            trainable.add("decoder")

    for name, model in models.items():
        requires_grad = name in trainable
        model.train(requires_grad)
        for param in model.parameters():
            param.requires_grad = requires_grad


def build_optimizer(models: Dict[str, nn.Module], args):
    params = [p for model in models.values() for p in model.parameters() if p.requires_grad]
    return AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)


def prepare_dataloaders(args) -> Tuple[DataLoader, Optional[DataLoader]]:
    raw_train = load_dataset(args.dataset_name, split=args.dataset_split)

    if args.validation_split_name:
        raw_val = load_dataset(args.dataset_name, split=args.validation_split_name)
    elif args.validation_split > 0:
        split_dataset = raw_train.train_test_split(test_size=args.validation_split, seed=args.seed)
        raw_train = split_dataset["train"]
        raw_val = split_dataset["test"]
    else:
        raw_val = None

    column_names = list(raw_train.column_names)
    image_column = resolve_column_name(column_names, args.image_column, ["image", "img"], "image")
    caption_column = None
    if args.stage == "diffusion":
        caption_column = resolve_column_name(column_names, args.caption_column, ["text", "caption", "prompt"], "caption")

    train_dataset = ImageTextDataset(raw_train, image_column=image_column, caption_column=caption_column, resolution=args.resolution)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_examples,
        drop_last=True,
    )

    val_loader = None
    if raw_val is not None:
        val_dataset = ImageTextDataset(raw_val, image_column=image_column, caption_column=caption_column, resolution=args.resolution)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_examples,
            drop_last=False,
        )

    return train_loader, val_loader


def run_vae_step(batch, models, optimizer, scaler, args, device):
    images = batch["pixel_values"].to(device)
    latent_h = images.shape[-2] // 8
    latent_w = images.shape[-1] // 8
    noise = torch.randn(images.shape[0], 4, latent_h, latent_w, device=device)

    amp_enabled = args.use_amp and device.startswith("cuda")
    with autocast(device_type="cuda", enabled=amp_enabled):
        latents, mean, log_var = models["encoder"].encode_stats(images, noise)
        recon = models["decoder"](latents)
        recon_loss = F.mse_loss(recon, images)
        kl_loss = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
        loss = recon_loss + args.kl_weight * kl_loss

    loss_to_backward = loss / args.grad_accum_steps
    if scaler is not None:
        scaler.scale(loss_to_backward).backward()
    else:
        loss_to_backward.backward()

    return {
        "loss": loss.detach(),
        "recon_loss": recon_loss.detach(),
        "kl_loss": kl_loss.detach(),
    }


def run_diffusion_step(batch, models, optimizer, scaler, args, device, sampler, tokenizer):
    images = batch["pixel_values"].to(device)
    captions = batch["captions"]

    if args.cfg_dropout_prob > 0:
        captions = ["" if torch.rand(1).item() < args.cfg_dropout_prob else caption for caption in captions]

    tokenized = tokenizer.batch_encode_plus(
        captions,
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )
    tokens = tokenized.input_ids.to(device)

    latent_h = images.shape[-2] // 8
    latent_w = images.shape[-1] // 8
    encoder_noise = torch.randn(images.shape[0], 4, latent_h, latent_w, device=device)
    noise = torch.randn(images.shape[0], 4, latent_h, latent_w, device=device)
    timesteps = torch.randint(0, sampler.num_train_timesteps, (images.shape[0],), device=device, dtype=torch.long)
    amp_enabled = args.use_amp and device.startswith("cuda")

    with torch.set_grad_enabled(args.train_encoder):
        latents = models["encoder"](images, encoder_noise)

    latents = latents.detach() if not args.train_encoder else latents
    noisy_latents = add_noise_for_training(latents, noise, timesteps, sampler.alphas_cumprod)
    time_emb = get_time_embedding_batch(timesteps).to(device)

    with torch.set_grad_enabled(args.train_clip):
        context = models["clip"](tokens)
    context = context.detach() if not args.train_clip else context

    with autocast(device_type="cuda", enabled=amp_enabled):
        pred_noise = models["diffusion"](noisy_latents, context, time_emb)
        loss = F.mse_loss(pred_noise, noise)

    loss_to_backward = loss / args.grad_accum_steps
    if scaler is not None:
        scaler.scale(loss_to_backward).backward()
    else:
        loss_to_backward.backward()

    return {"loss": loss.detach()}


@torch.no_grad()
def evaluate_vae(val_loader, models, args, device):
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    num_batches = 0
    amp_enabled = args.use_amp and device.startswith("cuda")

    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= args.num_val_batches:
            break
        images = batch["pixel_values"].to(device)
        latent_h = images.shape[-2] // 8
        latent_w = images.shape[-1] // 8
        noise = torch.randn(images.shape[0], 4, latent_h, latent_w, device=device)

        with autocast(device_type="cuda", enabled=amp_enabled):
            latents, mean, log_var = models["encoder"].encode_stats(images, noise)
            recon = models["decoder"](latents)
            recon_loss = F.mse_loss(recon, images)
            kl_loss = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
            loss = recon_loss + args.kl_weight * kl_loss

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        num_batches += 1

    if num_batches == 0:
        return None

    return {
        "val/loss": total_loss / num_batches,
        "val/recon_loss": total_recon / num_batches,
        "val/kl_loss": total_kl / num_batches,
    }


@torch.no_grad()
def evaluate_diffusion(val_loader, models, args, device, sampler, tokenizer):
    total_loss = 0.0
    num_batches = 0
    amp_enabled = args.use_amp and device.startswith("cuda")

    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= args.num_val_batches:
            break

        images = batch["pixel_values"].to(device)
        captions = batch["captions"]
        tokenized = tokenizer.batch_encode_plus(
            captions,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        tokens = tokenized.input_ids.to(device)

        latent_h = images.shape[-2] // 8
        latent_w = images.shape[-1] // 8
        encoder_noise = torch.randn(images.shape[0], 4, latent_h, latent_w, device=device)
        noise = torch.randn(images.shape[0], 4, latent_h, latent_w, device=device)
        timesteps = torch.randint(0, sampler.num_train_timesteps, (images.shape[0],), device=device, dtype=torch.long)
        latents = models["encoder"](images, encoder_noise)
        noisy_latents = add_noise_for_training(latents, noise, timesteps, sampler.alphas_cumprod)
        time_emb = get_time_embedding_batch(timesteps).to(device)
        context = models["clip"](tokens)

        with autocast(device_type="cuda", enabled=amp_enabled):
            pred_noise = models["diffusion"](noisy_latents, context, time_emb)
            loss = F.mse_loss(pred_noise, noise)

        total_loss += loss.item()
        num_batches += 1

    if num_batches == 0:
        return None

    return {"val/loss": total_loss / num_batches}


def set_eval_mode(models: Dict[str, nn.Module]):
    prev_modes = {}
    for name, model in models.items():
        prev_modes[name] = model.training
        model.eval()
    return prev_modes


def restore_train_mode(models: Dict[str, nn.Module], prev_modes: Dict[str, bool]):
    for name, model in models.items():
        model.train(prev_modes[name])


def save_tensor_as_image(image_tensor: torch.Tensor, output_path: Path):
    image = image_tensor.detach().clamp(0, 255).to(torch.uint8).cpu().numpy()
    Image.fromarray(image).save(output_path)


@torch.no_grad()
def save_vae_reconstructions(val_loader, models, args, device, global_step: int):
    sample_dir = Path(args.output_dir) / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    batch = next(iter(val_loader), None)
    if batch is None:
        return []

    images = batch["pixel_values"].to(device)
    latent_h = images.shape[-2] // 8
    latent_w = images.shape[-1] // 8
    noise = torch.randn(images.shape[0], 4, latent_h, latent_w, device=device)
    latents, _, _ = models["encoder"].encode_stats(images, noise)
    recon = models["decoder"](latents)
    recon = ((recon + 1.0) * 127.5).permute(0, 2, 3, 1)
    originals = ((images + 1.0) * 127.5).permute(0, 2, 3, 1)

    saved_paths = []
    max_items = min(args.num_sample_prompts, recon.shape[0])
    for idx in range(max_items):
        original_path = sample_dir / f"step_{global_step}_original_{idx}.png"
        recon_path = sample_dir / f"step_{global_step}_recon_{idx}.png"
        save_tensor_as_image(originals[idx], original_path)
        save_tensor_as_image(recon[idx], recon_path)
        saved_paths.extend([str(original_path), str(recon_path)])
    return saved_paths


@torch.no_grad()
def save_diffusion_samples(models, args, device, tokenizer, global_step: int):
    sample_dir = Path(args.output_dir) / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    prompts = args.sample_prompts or [
        "a cute pokemon creature",
        "a watercolor landscape",
        "a futuristic city at sunset",
        "a portrait illustration",
    ]
    prompts = prompts[: args.num_sample_prompts]
    saved_paths = []
    model_bundle = {
        "clip": models["clip"],
        "encoder": models["encoder"],
        "decoder": models["decoder"],
        "diffusion": models["diffusion"],
    }
    for idx, prompt in enumerate(prompts):
        image = generate(
            prompt=prompt,
            uncond_prompt="",
            do_cfg=True,
            cfg_scale=7.5,
            sampler_name="ddpm",
            n_inference_steps=20,
            models=model_bundle,
            seed=args.seed + global_step + idx,
            device=device,
            idle_device=device,
            tokenizer=tokenizer,
        )
        path = sample_dir / f"step_{global_step}_sample_{idx}.png"
        Image.fromarray(image).save(path)
        saved_paths.append(str(path))
    return saved_paths


def optimizer_step(optimizer, scaler, args):
    if scaler is not None:
        scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(
        [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None],
        args.max_grad_norm,
    )
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def maybe_init_swanlab(args):
    if not args.use_swanlab:
        return None
    if swanlab is None:
        raise ImportError("SwanLab is not installed. Please install swanlab or disable --use_swanlab.")
    return swanlab.init(project=args.swanlab_project, config=vars(args))


def run_validation_and_sampling(
    val_loader,
    models,
    args,
    device,
    sampler,
    tokenizer,
    global_step: int,
    epoch: int,
    optimizer,
    scaler,
    best_val_loss: Optional[float],
):
    if val_loader is None:
        return best_val_loss

    prev_modes = set_eval_mode(models)
    try:
        if args.stage == "vae":
            val_metrics = evaluate_vae(val_loader, models, args, device)
        else:
            val_metrics = evaluate_diffusion(val_loader, models, args, device, sampler, tokenizer)

        if val_metrics is None:
            return best_val_loss

        current_val_loss = val_metrics["val/loss"]
        if best_val_loss is None or current_val_loss < best_val_loss:
            save_best_checkpoint(
                Path(args.output_dir),
                global_step,
                epoch,
                models,
                optimizer,
                scaler if scaler.is_enabled() else None,
                args,
                metric_name="val/loss",
                metric_value=current_val_loss,
            )
            best_val_loss = current_val_loss

        if args.sample_every > 0 and global_step % args.sample_every == 0:
            if args.stage == "vae":
                saved_paths = save_vae_reconstructions(val_loader, models, args, device, global_step)
            else:
                saved_paths = save_diffusion_samples(models, args, device, tokenizer, global_step)
            val_metrics["samples/count"] = len(saved_paths)

        if args.use_swanlab and swanlab is not None:
            swanlab.log(val_metrics, step=global_step)
    finally:
        restore_train_mode(models, prev_modes)

    return best_val_loss


def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device
    output_dir = Path(args.output_dir)

    models = build_models(device)
    maybe_load_pretrained_models(models, args, device)
    set_trainable(models, args)
    optimizer = build_optimizer(models, args)
    scaler = GradScaler("cuda", enabled=args.use_amp and device.startswith("cuda"))

    start_epoch = 0
    global_step = 0
    best_val_loss = None
    if args.resume_from:
        start_epoch, global_step = load_checkpoint(args.resume_from, models, optimizer, scaler, device)

    train_loader, val_loader = prepare_dataloaders(args)
    tokenizer = maybe_load_tokenizer(args.tokenizer_name) if args.stage == "diffusion" else None
    sampler = DDPMSampler(torch.Generator(device=device), num_training_steps=args.num_train_timesteps)
    maybe_init_swanlab(args)

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, args.epochs):
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for batch_idx, batch in enumerate(progress):
            if args.stage == "vae":
                metrics = run_vae_step(batch, models, optimizer, scaler, args, device)
            else:
                metrics = run_diffusion_step(batch, models, optimizer, scaler, args, device, sampler, tokenizer)

            should_step = ((batch_idx + 1) % args.grad_accum_steps == 0)
            if should_step:
                optimizer_step(optimizer, scaler if scaler.is_enabled() else None, args)
                global_step += 1

                progress.set_postfix(loss=f"{metrics['loss'].item():.4f}")

                if args.use_swanlab and swanlab is not None and global_step % args.log_every == 0:
                    log_data = {"train/loss": metrics["loss"].item(), "train/lr": optimizer.param_groups[0]["lr"]}
                    if "recon_loss" in metrics:
                        log_data["train/recon_loss"] = metrics["recon_loss"].item()
                        log_data["train/kl_loss"] = metrics["kl_loss"].item()
                    swanlab.log(log_data, step=global_step)

                if val_loader is not None and args.validate_every > 0 and global_step % args.validate_every == 0:
                    best_val_loss = run_validation_and_sampling(
                        val_loader,
                        models,
                        args,
                        device,
                        sampler,
                        tokenizer,
                        global_step,
                        epoch,
                        optimizer,
                        scaler,
                        best_val_loss,
                    )

                if global_step % args.save_every == 0:
                    save_checkpoint(output_dir, global_step, epoch, models, optimizer, scaler if scaler.is_enabled() else None, args)

                if args.max_train_steps is not None and global_step >= args.max_train_steps:
                    if val_loader is not None and args.validate_every <= 0:
                        best_val_loss = run_validation_and_sampling(
                            val_loader,
                            models,
                            args,
                            device,
                            sampler,
                            tokenizer,
                            global_step,
                            epoch,
                            optimizer,
                            scaler,
                            best_val_loss,
                        )
                    save_checkpoint(output_dir, global_step, epoch, models, optimizer, scaler if scaler.is_enabled() else None, args)
                    return

        if val_loader is not None:
            best_val_loss = run_validation_and_sampling(
                val_loader,
                models,
                args,
                device,
                sampler,
                tokenizer,
                global_step,
                epoch,
                optimizer,
                scaler,
                best_val_loss,
            )
        save_checkpoint(output_dir, global_step, epoch, models, optimizer, scaler if scaler.is_enabled() else None, args)


if __name__ == "__main__":
    main()
