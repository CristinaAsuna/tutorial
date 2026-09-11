"""关卡 1：实现冻结视觉塔，并丢弃 CLS 后投影 patch features。"""
import torch


def vision_to_llm_tokens(vision_encoder, projector, pixel_values: torch.Tensor) -> torch.Tensor:
    # TODO: vision_encoder(pixel_values) gives (B, 1+N_patch, D_vision).
    # TODO: retain [:, 1:, :] and return projector(...) with D_llm.
    raise NotImplementedError("完成后可对照 reference_llava.LlavaForConditionalGeneration.encode_images")
