"""
cnn_torch.py
============
使用标准 `torch.nn.Conv2d` 的简易 CNN，供与 HandConvNet 对比。
此版本修正了 **回归任务** 输出尺寸不匹配 100×200 的问题：
    • 计算图像经过 3 次 MaxPool2d(2) 后的空间尺寸 ≈ (12, 25)
    • 采用 3 层 `ConvTranspose2d`(kernel_size=3, stride=2, padding=1, output_padding=1)
      可将尺寸精确放大 8 倍 → (96, 200)
    • 随后 `nn.Upsample(size=(100, 200))` 直接补齐到目标大小，确保
      输出 tensor 与 ground-truth 形状一致，避免广播报错。
"""

from __future__ import annotations

from typing import List

import torch
from torch import nn


class SimpleCNN(nn.Module):
    """
    Parameters
    ----------
    task : "classification" | "regression"
    num_classes : 仅在 classification 下使用
    conv_channels : 每个卷积层输出通道数列表
    """

    def __init__(
        self,
        *,
        task: str = "classification",
        num_classes: int = 3,
        conv_channels: List[int] | None = None,
    ):
        super().__init__()
        assert task in {"classification", "regression"}
        self.task = task

        conv_channels = conv_channels or [32, 64, 128]

        # ---------- 下采样特征提取 ---------- #
        layers: list[nn.Module] = []
        in_c = 3
        for out_c in conv_channels:
            layers += [
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # ↓  1/2
            ]
            in_c = out_c
        self.features = nn.Sequential(*layers)  # H,W 将被整体缩小 8 倍

        # ---------- 任务分支 ---------- #
        if task == "classification":
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_c, num_classes),
            )
        else:  # regression
            # 3 次 ConvTranspose2d → 尺寸 ×8 (= 100/12 ≈ 8, 200/25 = 8)
            self.reg_head = nn.Sequential(
                nn.ConvTranspose2d(
                    in_c, 128, kernel_size=3, stride=2, padding=1, output_padding=1
                ),  # ×2
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
                ),  # ×2
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
                ),  # ×2  共 ×8
                nn.ReLU(inplace=True),
                # 可能出现 96×200，最后一次双线性插值精确补齐 100×200
                nn.Upsample(size=(100, 200), mode="bilinear", align_corners=False),
                nn.Conv2d(32, 3, kernel_size=1),
                nn.Sigmoid(),
            )

        # ---------- 参数初始化 ---------- #
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        if self.task == "classification":
            return self.classifier(x)
        return self.reg_head(x)