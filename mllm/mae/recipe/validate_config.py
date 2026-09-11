import argparse, json
from pathlib import Path

p = argparse.ArgumentParser(description="Validate the MAE paper-recipe contract.")
p.add_argument("--config", default=Path(__file__).with_name("config.json"))
args = p.parse_args()
cfg = json.loads(Path(args.config).read_text())
assert cfg["model"]["mask_ratio"] == 0.75 and cfg["optimization"]["optimizer"] == "adamw"
print("MAE recipe config valid; data exists:", Path(cfg["data_root"]).is_dir())
