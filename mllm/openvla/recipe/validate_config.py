"""Validate metadata only: no downloads, training, weight loading or robot control."""
import argparse
import json
from pathlib import Path


def validate(cfg):
    data, model = cfg["dataset"], cfg["model"]
    if cfg["paper"] != "OpenVLA-2024-original" or cfg["implementation"] != "external_official":
        raise ValueError("recipe targets the original external OpenVLA implementation")
    if data["format"] != "RLDS" or data["action_dim"] != 7:
        raise ValueError("expected RLDS and 7-dimensional standardized actions")
    mask = data["normalization_mask"]
    if len(mask) != data["action_dim"] or any(type(value) is not bool for value in mask):
        raise ValueError("normalization_mask must have one boolean per action dimension")
    if data["normalization"] != "training_q01_q99" or not data["statistics_key"]:
        raise ValueError("explicit training quantiles/statistics key required")
    if model["action_horizon"] != 1 or model["tokenizer_edges"] != 256:
        raise ValueError("original single-step action-token contract required")
    if model["vision"] != ["DINOv2", "SigLIP"] or not model["predict_stop_token"]:
        raise ValueError("expected fused vision and default EOS supervision")
    if not cfg["training"]["stats_from_train_split_only"]:
        raise ValueError("do not fit normalization on evaluation data")
    if not model["checkpoint"]:
        raise ValueError("external checkpoint identifier required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    validate(cfg)
    print("OpenVLA recipe metadata valid; dataset exists:", Path(cfg["dataset"]["root"]).is_dir())
    print("External weights/data/training/evaluation NOT validated.")


if __name__ == "__main__":
    main()
