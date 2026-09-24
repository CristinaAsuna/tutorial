import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser(description="Validate InstructBLIP recipe metadata.")
p.add_argument("--config", default=Path(__file__).with_name("config.json"))
args = p.parse_args()
cfg = json.loads(Path(args.config).read_text())
assert cfg["backbones"]["frozen"] and cfg["qformer"]["instruction_aware"]
assert cfg["qformer"]["query_only_visual_cross_attention"]
print("InstructBLIP recipe config valid; data path exists:", Path(cfg["data"]["instruction_datasets"]).is_dir())
