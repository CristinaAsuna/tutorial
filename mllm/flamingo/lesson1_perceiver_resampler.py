"""关卡 1：把每张图的 patch features 压缩为固定数量 visual latents。"""
import torch


def resample(visual_features: torch.Tensor, latents: torch.Tensor):
    # TODO: 用 learnable latents 作 Query、视觉 patch 作 KV；输出 (B,M,R,D)。
    raise NotImplementedError("参考 reference_flamingo.PerceiverResampler")
