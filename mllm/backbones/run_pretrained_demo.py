"""Native local image inference and reusable feature/KV checks. Explicit download is separate."""
import argparse
import json
import time
import torch
from PIL import Image
from edu_core.backbones import load_backbone, prefix_encoding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path")
    parser.add_argument("--image", default=None)
    parser.add_argument("--question", default="What color is the square?")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    args = parser.parse_args()
    from pathlib import Path
    image_path = Path(args.image) if args.image else Path(__file__).with_name("assets")/"red_square.png"
    image = Image.open(image_path).convert("RGB")
    torch.set_num_threads(2)
    started = time.perf_counter()
    model = load_backbone("smolvlm", model_path=args.model_path, frozen=True)
    reply = model.native_chat(image, args.question, args.max_new_tokens)
    assert reply.strip(), "native model produced no text"
    import numpy as np
    pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1)[None].float()/255
    with torch.no_grad():
        raw = model.encode_vision(pixels)
        projected = model.encode_vision(pixels, projected=True)
        ids = model.tokenizer("Describe the square.", return_tensors="pt")["input_ids"]
        text = model.embed_text(ids)
        prefix = torch.cat((projected.tokens, text), 1)
        valid = torch.ones(prefix.shape[:2], dtype=torch.bool)
        cache = prefix_encoding(model, prefix, valid)
    assert len(cache.key_values) == model.spec.num_layers
    assert cache.key_values[0][0].shape[1] == model.spec.num_kv_heads
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    differentiable_prefix = prefix.detach().requires_grad_(True)
    differentiable_cache = prefix_encoding(model, differentiable_prefix, valid)
    key, value = differentiable_cache.key_values[-1]
    (key.square().mean()+value.square().mean()).backward()
    assert torch.isfinite(differentiable_prefix.grad).all() and differentiable_prefix.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters())
    print(json.dumps({**model.metadata, "native_reply": reply, "raw_tokens": list(raw.tokens.shape),
                      "projected_tokens": list(projected.tokens.shape), "kv_layers": len(cache.key_values),
                      "kv_first_shape": list(cache.key_values[0][0].shape), "dtype": "float32", "device": "cpu",
                      "kv_input_gradient": True,
                      "seconds": round(time.perf_counter()-started, 2)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
