"""Validate recipe metadata only; no robot data/weights/training are loaded."""
import argparse
import json
from pathlib import Path


def validate(cfg):
    if cfg["paper"] != "pi0-2024-original" or cfg["model"]["pi05"]:
        raise ValueError("this course targets original pi0, not pi0.5")
    flow = cfg["flow"]
    if (flow["noise_time"], flow["data_time"], flow["target"]) != (1, 0, "noise_minus_actions"):
        raise ValueError("time direction and velocity must follow the same convention")
    if flow["euler_steps"] <= 0 or not flow["prefix_cache"]:
        raise ValueError("positive integration steps and prefix KV required")
    if not 0 < cfg["actions"]["execute_steps"] <= cfg["model"]["action_horizon"]:
        raise ValueError("execution chunk must fit prediction horizon")
    if not cfg["dataset"]["split_by_episode"] or not cfg["dataset"]["camera_order"]:
        raise ValueError("episode split and explicit camera order required")
    if not cfg["dataset"]["state_and_action_units_required"]:
        raise ValueError("state/action units cannot be inferred from dimension alone")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    validate(cfg)
    print("pi0 recipe metadata valid; dataset exists:", Path(cfg["dataset"]["root"]).is_dir())
    print("Official PaliGemma/openpi weights, real data and rollout metrics NOT validated.")


if __name__ == "__main__":
    main()
