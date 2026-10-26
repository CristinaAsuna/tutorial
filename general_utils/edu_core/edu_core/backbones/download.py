"""Explicit public download; ordinary model loading is local-only."""
import argparse
import json
from pathlib import Path

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"


def download_model(model_id=MODEL_ID, revision="main", cache_dir=None):
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi().model_info(model_id, revision=revision)
    resolved = info.sha
    folder = snapshot_download(model_id, revision=resolved, cache_dir=cache_dir,
                               allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt", "README.md"],
                               ignore_patterns=["onnx/*", "*.onnx", "*.bin"], max_workers=2)
    metadata = {"model_id": model_id, "revision": resolved, "snapshot": folder}
    # Metadata lives in the cache; no machine-specific path is committed to the repository.
    record = Path(folder).parent.parent/"tutorial-download.json"
    record.write_text(json.dumps(metadata, indent=2)+"\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cache-dir")
    args = parser.parse_args()
    print(json.dumps(download_model(args.model_id, args.revision, args.cache_dir), indent=2))


if __name__ == "__main__":
    main()
