import torch
from torchvision.utils import save_image


def denormalize_tanh(images):
    """将训练使用的 [-1,1] 图像转换为可保存的 [0,1] 图像。"""
    return ((images + 1.0) * 0.5).clamp(0.0, 1.0)


def save_reconstruction_grid(images, reconstructions, path, sample_count=8):
    count = min(sample_count, images.shape[0])
    comparison = torch.cat([images[:count], reconstructions[:count]], dim=0)
    save_image(denormalize_tanh(comparison), path, nrow=count)
