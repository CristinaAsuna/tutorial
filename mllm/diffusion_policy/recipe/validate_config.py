"""Metadata validation only; no simulator, Zarr loader, trainer or benchmark execution."""
import argparse
import json
from pathlib import Path


def validate(cfg):
    data, policy = cfg["dataset"], cfg["policy"]
    if cfg["paper"] != "DiffusionPolicy-RSS2023" or cfg["implementation"] != "external_official":
        raise ValueError("expected external RSS 2023 Diffusion Policy")
    to, tp, ta = policy["obs_horizon"], policy["pred_horizon"], policy["action_horizon"]
    if any(type(x) is not int or x <= 0 for x in (to, tp, ta)) or to-1+ta > tp:
        raise ValueError("positive horizons must satisfy To-1+Ta <= Tp")
    if set(policy["backbones"]) != {"cnn", "transformer"}:
        raise ValueError("both backbone recipes required")
    if tp % 4:
        raise ValueError("official multi-level CNN recipe needs horizon divisible by 4")
    if policy["execution_offset"] != "obs_horizon_minus_one":
        raise ValueError("execution starts at current observation index To-1")
    if policy["prediction_type"] != "epsilon" or policy["beta_schedule"] != "squaredcos_cap_v2":
        raise ValueError("epsilon objective and square-cosine schedule required")
    if policy["diffusion_steps"] != 100 or not policy["ema"]:
        raise ValueError("this official recipe uses 100 diffusion steps and EMA")
    if policy["conditioning"] != "observations_only":
        raise ValueError("do not jointly diffuse future observations")
    if data["task"] != "pusht_image" or data["action_dim"] != 2:
        raise ValueError("sample config targets 2D image Push-T")
    if not data["split_by_episode"] or not cfg["training"]["stats_from_train_split_only"]:
        raise ValueError("episode split and training-only statistics required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    validate(cfg)
    print("Diffusion Policy recipe metadata valid; dataset exists:", Path(cfg["dataset"]["root"]).exists())
    print("External data/weights/training/rollouts NOT validated.")


if __name__ == "__main__":
    main()
