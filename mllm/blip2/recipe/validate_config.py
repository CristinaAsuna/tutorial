import argparse, json
from pathlib import Path

p = argparse.ArgumentParser(description="Validate BLIP-2 recipe metadata.")
p.add_argument("--config", default=Path(__file__).with_name("config.json"))
args = p.parse_args()
cfg = json.loads(Path(args.config).read_text())
assert cfg["vision_encoder"]["frozen"] and set(cfg["stage1_losses"]) == {"itc", "itm", "itg"}
print("BLIP-2 recipe config valid; stage-1 data exists:", Path(cfg["data"]["stage1"]).is_dir())
