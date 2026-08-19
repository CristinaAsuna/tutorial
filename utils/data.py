from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FlatImageDataset(Dataset):
    """读取平铺图片目录；适合 CelebA-HQ 这类没有类别子文件夹的数据。"""

    extensions = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(self, root, transform):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {self.root}")
        self.paths = sorted(
            path for path in self.root.rglob("*") if path.suffix.lower() in self.extensions
        )
        if not self.paths:
            raise RuntimeError(f"No supported image files found in: {self.root}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def build_image_transform(image_size, value_range="tanh"):
    """创建统一图像预处理。tanh 输出 [-1,1]，sigmoid 输出 [0,1]。"""
    steps = [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ]
    if value_range == "tanh":
        steps.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    elif value_range != "sigmoid":
        raise ValueError("value_range must be 'tanh' or 'sigmoid'.")
    return transforms.Compose(steps)
