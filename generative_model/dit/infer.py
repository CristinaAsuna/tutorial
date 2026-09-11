"""Sample a trained DiT checkpoint for DDPM, Flow Matching, or VP-SDE."""

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dit.pipeline import build_pipeline
from utils.checkpoint import read_checkpoint
from utils.dit import DiT, DiTConfig
from utils.inference import save_generated_grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", choices=["ddpm", "flow_matching", "vp_sde"], required=True)
    parser.add_argument("--steps", type=int, default=None, help="Flow/VP-SDE sampling steps")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    names = {"ddpm": "ddpm_latest.pt", "flow_matching": "flow_matching_latest.pt", "vp_sde": "score_sde_latest.pt"}
    checkpoint_path = Path(__file__).with_name("checkpoints") / args.objective / names[args.objective]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = read_checkpoint(checkpoint_path, device)

    model_data = checkpoint["extra"]["model_config"]["config"]
    config = DiTConfig(**model_data)
    model = DiT(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    train_config = checkpoint["config"]
    if args.objective == "ddpm":
        pipeline = build_pipeline("ddpm")
        samples = pipeline.sample(model, args.batch_size, (config.in_channels, config.image_size, config.image_size), device)
    elif args.objective == "flow_matching":
        pipeline = build_pipeline("flow_matching", time_scale=train_config["time_scale"])
        steps = args.steps or train_config["sample_steps"]
        samples = pipeline.sample(model, args.batch_size, (config.in_channels, config.image_size, config.image_size), steps, device)
    else:
        pipeline = build_pipeline("vp_sde", time_scale=train_config["time_scale"], beta_min=train_config["beta_min"], beta_max=train_config["beta_max"], sde_eps=train_config["sde_eps"])
        steps = args.steps or train_config["sample_steps"]
        samples = pipeline.sample_euler_maruyama(model, args.batch_size, (config.in_channels, config.image_size, config.image_size), steps, device)

    output = checkpoint_path.with_name("samples.png")
    save_generated_grid(samples.cpu(), output, nrow=4)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
