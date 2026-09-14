"""仅校验 V-JEPA recipe 合同；不启动大规模训练。"""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--config", default=Path(__file__).with_name("config.json"))
args = p.parse_args()
cfg = json.loads(Path(args.config).read_text())
assert cfg["objective"]["target"] == "ema_teacher_latent"
assert cfg["objective"]["pixel_reconstruction"] is False
assert cfg["masking"]["context_is_complement"] is True
assert len(cfg["masking"]["target_block"]) == 3
print("V-JEPA recipe config valid; data exists:", Path(cfg["data_root"]).is_dir())
