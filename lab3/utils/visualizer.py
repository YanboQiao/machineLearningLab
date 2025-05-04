"""
visualizer.py
=============
绘制 Loss / Metric 曲线 & 保存图片示例
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
from PIL import Image

from .transforms import denormalize


def plot_history(
    history: Dict[str, List[float]],
    save_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """
    根据 engine.train 返回的 history 字典绘制曲线
    """
    epochs = history["epoch"]

    fig, ax1 = plt.subplots()
    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    ax2 = ax1.twinx()
    ax2.plot(epochs, history["train_metric"], "--", label="Train Metric")
    ax2.plot(epochs, history["val_metric"], "--", label="Val Metric")
    ax2.set_ylabel("Metric")
    ax2.legend(loc="lower right")

    fig.tight_layout()
    if save_path is not None:
        plt.savefig(str(save_path))
    if show:
        plt.show()
    plt.close(fig)


def save_image_triplet(
    fog: torch.Tensor,
    pred: torch.Tensor,
    gt: torch.Tensor,
    out_path: str | Path,
) -> None:
    """
    将 (3,H,W) 3 张 tensor 拼接保存
    """
    with torch.no_grad():
        a = denormalize(fog).permute(1, 2, 0).cpu()
        b = denormalize(pred).permute(1, 2, 0).cpu()
        c = denormalize(gt).permute(1, 2, 0).cpu()

        concat = torch.cat([a, b, c], dim=1).numpy()
        im = Image.fromarray(concat)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        im.save(out_path)