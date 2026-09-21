import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser(description="Validate Flamingo recipe metadata.")
p.add_argument("--config", default=Path(__file__).with_name("config.json"))
args = p.parse_args()
cfg = json.loads(Path(args.config).read_text())
assert cfg["vision_encoder"]["frozen"] and cfg["language_model"]["frozen"]
assert cfg["connector"]["zero_init_tanh_gates"] and cfg["objective"]["interleaved_media_text"]
print("Flamingo recipe config valid; M3W path exists:", Path(cfg["data"]["interleaved_web"]).is_dir())
