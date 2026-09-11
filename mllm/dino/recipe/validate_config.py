import argparse, json
from pathlib import Path

p = argparse.ArgumentParser(description="Validate DINO/iBOT recipe metadata.")
p.add_argument("--config", default=Path(__file__).with_name("config.json"))
args = p.parse_args()
cfg = json.loads(Path(args.config).read_text())
assert cfg["augmentation"]["global_crops"] == 2 and cfg["runtime"]["center_sync"] == "all_reduce"
print("DINO recipe config valid; data exists:", Path(cfg["data_root"]).is_dir())
