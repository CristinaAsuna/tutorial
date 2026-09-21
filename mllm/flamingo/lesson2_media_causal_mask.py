"""关卡 2：文本只能读取左侧已经出现的图像。"""
import torch


def build_media_mask(input_ids: torch.Tensor, image_token_index: int, num_images: int, latents_per_image: int):
    # TODO: image sentinel 做 cumsum；每个文本位置只允许 image_index < seen_images。
    raise NotImplementedError("参考 FlamingoForConditionalGeneration.build_media_attention_mask")
