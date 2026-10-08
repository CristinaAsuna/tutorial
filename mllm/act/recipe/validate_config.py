import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser(description="Validate ACT recipe metadata.")
p.add_argument("--config", default=Path(__file__).with_name("config.json"))
args = p.parse_args()
cfg = json.loads(Path(args.config).read_text())
assert cfg["actions"]["representation"] == "leader_target_absolute_joint_positions"
assert cfg["policy"]["inference_style_latent"] == "zero"
assert cfg["policy"]["temporal_ensemble"].startswith("exponential")
print("ACT recipe config valid; dataset path exists:", Path(cfg["runtime"]["dataset_root"]).is_dir())
