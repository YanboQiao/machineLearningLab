"""
transforms.py
=============
集中放置数据增强 / 预处理，方便统一修改。
"""

from __future__ import annotations

from typing import Tuple

import torch
from PIL import Image
from torchvision import transforms as T

IMG_SIZE: Tuple[int, int] = (100, 200)  # (H, W)


def get_default_transform() -> T.Compose:
    """
    统一 Resize→ToTensor→Normalize(0-1)。
    若后续需要数据增强，可在此处插入 RandomHorizontalFlip 等。
    """
    return T.Compose(
        [
            T.Resize(IMG_SIZE, interpolation=Image.BILINEAR),
            T.ToTensor(),          # 自动把 0-255 → 0-1
        ]
    )


# -------------- 反归一化 -------------- #
def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    将 0-1 tensor → 0-255 uint8，便于保存/显示
    """
    tensor = tensor.clamp(0, 1) * 255.0
    return tensor.to(dtype=torch.uint8)
