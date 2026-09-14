import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser(description="Validate the I-JEPA paper-recipe contract.")
p.add_argument("--config", default=Path(__file__).with_name("config.json"))
args = p.parse_args()
cfg = json.loads(Path(args.config).read_text())
assert cfg["optimization"]["optimizer"] == "adamw"
assert cfg["masking"]["num_target_blocks"] > 0
assert cfg["optimization"]["ema_momentum"][1] == 1.0
print("I-JEPA recipe config valid; data exists:", Path(cfg["data_root"]).is_dir())
