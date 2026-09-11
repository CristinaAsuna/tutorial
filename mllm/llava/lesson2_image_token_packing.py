"""关卡 2：把单个 IMAGE_TOKEN_INDEX 展开成连续的视觉 token embedding。"""
import torch


def pack_one_image(input_ids: torch.Tensor, image_features: torch.Tensor):
    # TODO: validate exactly one -200 per sample; replace it with N_patch features.
    # TODO: pad the resulting embedding sequences on the right.
    raise NotImplementedError("参考 reference_llava.pack_multimodal_inputs")
