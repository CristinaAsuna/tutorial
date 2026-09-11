import argparse, json
from pathlib import Path

p = argparse.ArgumentParser(description="Validate LLaVA recipe metadata.")
p.add_argument("--config", default=Path(__file__).with_name("config.json"))
args = p.parse_args()
cfg = json.loads(Path(args.config).read_text())
assert cfg["projector"] == "mlp2x_gelu" and cfg["vision_encoder"]["feature_select"] == "patch"
print("LLaVA recipe config valid; alignment data exists:", Path(cfg["data"]["alignment"]).is_dir())
