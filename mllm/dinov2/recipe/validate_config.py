"""Validate metadata; missing data never implies successful training."""
import argparse
import json
from pathlib import Path
from mllm.dino.recipe_validation import validate_recipe

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'))
p.add_argument('--require-data',action='store_true')
a=p.parse_args()
print(json.dumps(validate_recipe(json.loads(a.config.read_text()),a.require_data),ensure_ascii=False))
