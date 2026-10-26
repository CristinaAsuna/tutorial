"""Optional foundation-model adapters. Importing the offline path never imports transformers."""
from .types import Backbone, DecoderSpec, LanguageOutput, PrefixEncoding, VisionFeatures, prefix_encoding, require_capabilities, valid_positions
from .tiny import TinyBackbone


def load_backbone(name="tiny", *, model_path=None, device="cpu", dtype=None, frozen=False):
    import torch
    if name == "tiny":
        model = TinyBackbone()
    elif name == "smolvlm":
        from .smolvlm import SmolVLMBackbone
        model = SmolVLMBackbone(model_path=model_path, dtype=dtype or torch.float32)
    else:
        raise ValueError(f"unknown backbone {name!r}; choose tiny or smolvlm")
    model.to(device=device, dtype=dtype or torch.float32)
    if frozen:
        model.requires_grad_(False).eval()
    return model
