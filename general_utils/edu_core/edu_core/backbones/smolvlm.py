"""Local-only SmolVLM adapter, tested against transformers 5.15.

Native chat uses the official processor. Component mode exposes raw/projected
vision features and differentiable per-layer prefix KV, without changing HF code.
"""
import json
from pathlib import Path
import torch
from torch import nn
from .download import MODEL_ID
from .types import DecoderSpec, LanguageOutput, VisionFeatures, valid_positions


def cached_snapshot():
    from huggingface_hub.constants import HF_HUB_CACHE
    record = Path(HF_HUB_CACHE)/("models--"+MODEL_ID.replace("/", "--"))/"tutorial-download.json"
    if not record.is_file():
        raise FileNotFoundError("Run python3 -m edu_core.backbones.download explicitly, or supply model_path")
    metadata = json.loads(record.read_text())
    folder = Path(metadata["snapshot"])
    if not folder.is_dir():
        raise FileNotFoundError(f"cached snapshot missing: {folder}")
    return folder, metadata


class SmolVLMBackbone(nn.Module):
    capabilities = frozenset({"vision", "text", "generation", "block_mask", "layer_kv", "native_chat"})

    def __init__(self, model_path=None, dtype=torch.float32):
        super().__init__()
        from transformers import AutoProcessor, AutoModelForImageTextToText
        if model_path is None:
            model_path, self.metadata = cached_snapshot()
        else:
            self.metadata = {"model_id": MODEL_ID, "snapshot": str(model_path), "revision": Path(model_path).name}
        self.processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
        # The 256M checkpoint declares Idefics3, despite the SmolVLM product name.
        # Respect its config instead of forcing another compatible-looking class.
        self.model, loading = AutoModelForImageTextToText.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False,
            dtype=dtype, attn_implementation="eager", output_loading_info=True)
        if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise RuntimeError(f"checkpoint was not loaded exactly: {loading}")
        config = self.model.config.text_config
        if config.model_type != "llama":
            raise ValueError("this adapter's layer/KV geometry requires the supported Llama text backbone")
        if config.rope_parameters.get("rope_type", "default") != "default":
            raise ValueError("scaled RoPE needs an explicitly supported expert rotary adapter")
        self.spec = DecoderSpec(config.hidden_size, config.num_hidden_layers, config.num_attention_heads,
                                config.num_key_value_heads, config.head_dim, config.vocab_size,
                                config.rope_parameters["rope_theta"])
        self.vision_dim = self.model.config.vision_config.hidden_size

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    @property
    def text_model(self):
        return self.model.model.text_model

    def embed_text(self, ids):
        return self.model.get_input_embeddings()(ids)

    def encode_vision(self, images, *, projected=False):
        # Component contract: float RGB [0,1], one square image per row; no hidden crop explosion.
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must be float RGB (B,3,H,W) in [0,1]")
        if not images.is_floating_point() or not torch.isfinite(images).all() or images.amin() < 0 or images.amax() > 1:
            raise ValueError("component images must be finite float RGB in [0,1]; do not pass uint8 or normalized pixels")
        vision_size = self.model.config.vision_config.image_size
        pixels = torch.nn.functional.interpolate(images, (vision_size, vision_size), mode="bilinear", align_corners=False)
        # SigLIP normalization, matching the official processor's mean/std.
        mean = pixels.new_tensor(self.processor.image_processor.image_mean)[None, :, None, None]
        std = pixels.new_tensor(self.processor.image_processor.image_std)[None, :, None, None]
        pixels = (pixels-mean)/std
        raw = self.model.model.vision_model(pixels.to(self.model.dtype)).last_hidden_state
        tokens = self.model.model.connector(raw) if projected else raw
        side = round(tokens.shape[1]**.5)
        return VisionFeatures(tokens, torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device), (side, side))

    def language(self, embeddings, valid_mask, *, allow_mask=None, positions=None, use_cache=False, output_logits=True):
        if positions is None:
            positions = valid_positions(valid_mask)
        if allow_mask is None:
            # Native causal forward uses a 2D key-valid mask.
            mask = valid_mask.long()
        else:
            if allow_mask.shape != (len(embeddings), embeddings.shape[1], embeddings.shape[1]):
                raise ValueError("explicit allow_mask must be (B,L,L)")
            permitted = allow_mask & valid_mask[:, :, None] & valid_mask[:, None, :]
            mask = torch.zeros(permitted.shape, dtype=embeddings.dtype, device=embeddings.device)
            mask = mask.masked_fill(~permitted, torch.finfo(embeddings.dtype).min)[:, None]
        output = self.text_model(inputs_embeds=embeddings, attention_mask=mask, position_ids=positions, use_cache=use_cache)
        key_values = ()
        if use_cache:
            key_values = tuple((layer.keys, layer.values) for layer in output.past_key_values.layers)
        hidden = output.last_hidden_state
        return LanguageOutput(hidden, self.model.lm_head(hidden) if output_logits else None, key_values)

    @torch.no_grad()
    def generate(self, ids, max_new_tokens=4):
        self.eval()
        return self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                   max_new_tokens=max_new_tokens, do_sample=False)

    @torch.no_grad()
    def native_chat(self, image, question="Describe this image.", max_new_tokens=24):
        self.eval()
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        # Limit to one image crop for a reproducible inexpensive CPU test.
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt", do_image_splitting=False)
        device = next(self.parameters()).device
        inputs = inputs.to(device)
        generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new_tokens = generated[:, inputs["input_ids"].shape[1]:]
        return self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
