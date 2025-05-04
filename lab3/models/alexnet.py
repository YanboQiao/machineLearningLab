"""
alexnet.py
==========
改编自 torchvision.models.alexnet，但针对输入 (3,100,200)、
输出通道可选（分类/去雾）。
"""

from __future__ import annotations

import torch
from torch import nn


class AlexNet(nn.Module):
    def __init__(self, *, task: str = "classification", num_classes: int = 3):
        super().__init__()
        assert task in {"classification", "regression"}
        self.task = task

        # -------- Feature Extractor -------- #
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  # 100x200 → 24x48
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # 12x23
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # 6x11
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # 3x5
        )

        if task == "classification":
            self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
        else:
            # 反卷积恢复到 (3,100,200)
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, output_padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, output_padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=4),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 3, kernel_size=1),
                nn.Sigmoid(),
            )

        # 初始化
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        if self.task == "classification":
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x
        return self.decoder(x)