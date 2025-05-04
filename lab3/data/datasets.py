"""
datasets.py
===========

实现两个 `torch.utils.data.Dataset` 子类：

1. VehicleDataset —— 三分类任务（car / bus / truck）
2. DefogDataset    —— 图片去雾回归任务

两者均支持：
    • 80/20 训练-测试自动划分
    • 默认将图片 Resize→(100, 200)，再转 tensor 并归一化到 0~1
    • 可通过 `as_numpy=True` 让 __getitem__ 返回 numpy.ndarray 而非 torch.Tensor
    • 随机打乱划分时固定随机种子，保证复现实验

此外提供一个辅助函数 `build_dataloaders`，方便快速获得 DataLoader。
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# --------------------------- 常量与默认预处理 --------------------------- #
IMG_SIZE: Tuple[int, int] = (100, 200)  # (height, width)，题目要求 “200 100” -> 宽 200, 高 100

_default_transform = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE, interpolation=Image.BILINEAR),
        transforms.ToTensor(),  # 同时完成 0‒255 → 0‒1 归一化
    ]
)


# --------------------------- 工具函数 --------------------------- #
def _split_indices(
    n_total: int, test_ratio: float = 0.2, seed: int = 42
) -> Tuple[List[int], List[int]]:
    """返回 train_idx, test_idx（已随机但可复现）"""
    indices = list(range(n_total))
    random.seed(seed)
    random.shuffle(indices)
    split = int(n_total * (1.0 - test_ratio))
    train_idx = indices[:split]
    test_idx = indices[split:]
    return train_idx, test_idx


def _pil_loader(path: Path) -> Image.Image:
    """默认的 PIL 读取函数；若 pillow-simd 可自动加速。"""
    return Image.open(path).convert("RGB")


# --------------------------- VehicleDataset --------------------------- #
class VehicleDataset(Dataset):
    """
    车辆三分类数据集

    目录结构
    ----------
    Datasets /
        viecleClassificationDataset /
            car   / *.jpg|png
            bus   / *.jpg|png
            truck / *.jpg|png

    示例
    ----
    >>> train_ds = VehicleDataset(root="Datasets/viecleClassificationDataset", train=True)
    >>> img, label = train_ds[0]        # img: (3,H,W) tensor, label: int64
    """

    _CLASSES = ("car", "bus", "truck")
    _EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        transform: Optional[Callable] = None,
        loader: Callable[[Path], Image.Image] = _pil_loader,
        test_ratio: float = 0.2,
        seed: int = 42,
        as_numpy: bool = False,
    ):
        super().__init__()
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"root directory {self.root} not found.")

        self.train = train
        self.transform = transform or _default_transform
        self.loader = loader
        self.as_numpy = as_numpy

        self.samples: List[Tuple[Path, int]] = []
        self._label_map = {name: idx for idx, name in enumerate(self._CLASSES)}

        # 遍历类别文件夹
        for cls_name in self._CLASSES:
            cls_dir = self.root / cls_name
            if not cls_dir.is_dir():
                raise FileNotFoundError(
                    f"Expected sub-directory {cls_dir} for class “{cls_name}”."
                )
            for path in cls_dir.rglob("*"):
                if path.suffix.lower() in self._EXTENSIONS:
                    self.samples.append((path, self._label_map[cls_name]))

        if len(self.samples) == 0:
            raise RuntimeError(f"No image files found under {self.root}.")

        # 划分 train / test
        n_total = len(self.samples)
        train_idx, test_idx = _split_indices(
            n_total=n_total, test_ratio=test_ratio, seed=seed
        )
        selected = train_idx if self.train else test_idx
        # 为保证 DataLoader 顺序稳定，用 sorted
        self.samples = sorted([self.samples[i] for i in selected], key=lambda x: x[0])

    # ==== 必须方法 ==== #
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = self.loader(img_path)
        img = self.transform(img)

        if self.as_numpy:
            img = img.numpy()
        return img, torch.tensor(label, dtype=torch.long)


# --------------------------- DefogDataset --------------------------- #
class DefogDataset(Dataset):
    """
    图片去雾数据集（配对回归）

    目录结构
    ----------
    Datasets /
        defogDataset /
            foggy   / xxx.png
            nofoggy / xxx.png        # 文件名需一一对应

    返回
    ----
    (fog_tensor, clear_tensor)  或  (fog_ndarray, clear_ndarray) 若 as_numpy=True
    """

    _EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        transform: Optional[Callable] = None,
        loader: Callable[[Path], Image.Image] = _pil_loader,
        test_ratio: float = 0.2,
        seed: int = 42,
        as_numpy: bool = False,
    ):
        super().__init__()
        self.root = Path(root)
        fog_dir = self.root / "foggy"
        clear_dir = self.root / "nofoggy"

        if not (fog_dir.is_dir() and clear_dir.is_dir()):
            raise FileNotFoundError(
                f"Expect sub-dirs `foggy` and `nofoggy` under {self.root}."
            )

        self.train = train
        self.transform = transform or _default_transform
        self.loader = loader
        self.as_numpy = as_numpy

        # 配对图像 path 列表
        fog_paths, clear_paths = self._match_pairs(fog_dir, clear_dir)
        self.pairs = list(zip(fog_paths, clear_paths))

        # 划分
        n_total = len(self.pairs)
        train_idx, test_idx = _split_indices(
            n_total=n_total, test_ratio=test_ratio, seed=seed
        )
        selected = train_idx if self.train else test_idx
        self.pairs = sorted([self.pairs[i] for i in selected], key=lambda x: x[0])

    # ==== 私有辅助 ==== #
    @staticmethod
    def _match_pairs(fog_dir: Path, clear_dir: Path) -> Tuple[List[Path], List[Path]]:
        """对 fog → clear 按文件名（不含扩展名）做一一对应"""
        fog_dict = {
            p.stem: p
            for p in fog_dir.iterdir()
            if p.suffix.lower() in DefogDataset._EXTENSIONS
        }
        clear_dict = {
            p.stem: p
            for p in clear_dir.iterdir()
            if p.suffix.lower() in DefogDataset._EXTENSIONS
        }
        common_keys = sorted(set(fog_dict) & set(clear_dict))
        if not common_keys:
            raise RuntimeError("No matching file names between foggy/ and nofoggy/.")

        fog_list = [fog_dict[k] for k in common_keys]
        clear_list = [clear_dict[k] for k in common_keys]
        return fog_list, clear_list

    # ==== 必须方法 ==== #
    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        fog_path, clear_path = self.pairs[idx]
        fog_img = self.transform(self.loader(fog_path))
        clear_img = self.transform(self.loader(clear_path))

        if self.as_numpy:
            fog_img = fog_img.numpy()
            clear_img = clear_img.numpy()
        return fog_img, clear_img


# --------------------------- DataLoader 辅助 --------------------------- #
def build_dataloaders(
    dataset_cls,
    root: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle_train: bool = True,
    **dataset_kwargs,
) -> Tuple[DataLoader, DataLoader]:
    """
    快速为给定 Dataset 类生成 train_loader, test_loader

    Parameters
    ----------
    dataset_cls : VehicleDataset 或 DefogDataset
    root        : 数据集根目录
    batch_size  : DataLoader 批大小
    num_workers : DataLoader 线程
    shuffle_train : 是否在 train_loader 中打乱
    **dataset_kwargs : 其余传给 Dataset 的参数

    Returns
    -------
    train_loader, test_loader
    """
    train_ds = dataset_cls(root=root, train=True, **dataset_kwargs)
    test_ds = dataset_cls(root=root, train=False, **dataset_kwargs)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader