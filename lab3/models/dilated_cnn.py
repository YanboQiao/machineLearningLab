"""
dilated_cnn.py
==============
层叠空洞卷积 (Dilated Convolution) 网络，满足 **HDC**（Hierarchical
Dilated Convolution）设计原则：

    • dilation 递增：d₁ < d₂ < … < dₙ
    • 相邻 dilation 互质：gcd(dᵢ, dᵢ₊₁) = 1
      → 经典用例 1, 2, 5，即可涵盖大感受野又避免“棋盘空洞”。

支持两种任务：
    - task='classification' → 输出 (N, num_classes)
    - task='regression'     → 输出 (N, 3, 100, 200)（去雾）

"""

from __future__ import annotations

import math
from typing import List

import torch
from torch import nn


# ------------------------------------------------------------------ #
# HDC 合规性检查
# ------------------------------------------------------------------ #
def _verify_hdc(dilations: List[int]) -> None:
    """
    约束：
        1. 严格递增
        2. 相邻 dilation 的最大公约数 == 1
    典型序列，例如 [1,2,5] 或 [1,3,7] 都合法。
    """
    if any(d <= 0 for d in dilations):
        raise ValueError("dilation 必须为正整数")
    for a, b in zip(dilations, dilations[1:]):
        if not (b > a and math.gcd(a, b) == 1):
            raise ValueError(
                f"HDC 约束不满足：{dilations}；需递增且相邻互质，例如 1,2,5 或 1,3,7"
            )


# ------------------------------------------------------------------ #
# 网络实现
# ------------------------------------------------------------------ #
class DilatedCNN(nn.Module):
    """
    Parameters
    ----------
    task : 'classification' | 'regression'
    num_classes : 分类类别数
    base_channels : 第一层输出通道
    dilation_list : 满足 HDC 的 dilation 序列
    """

    IMG_H, IMG_W = 100, 200

    def __init__(
        self,
        *,
        task: str = "classification",
        num_classes: int = 3,
        base_channels: int = 32,
        dilation_list: List[int] | None = None,
    ):
        super().__init__()
        assert task in {"classification", "regression"}
        self.task = task

        dilation_list = dilation_list or [1, 2, 5]
        _verify_hdc(dilation_list)

        # --------- 特征提取 --------- #
        layers: list[nn.Module] = []
        in_c = 3
        for i, dil in enumerate(dilation_list):
            out_c = base_channels * (2**i)
            pad = dil  # kernel=3 时, pad = dilation 可保持分辨率
            layers += [
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=pad, dilation=dil),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            ]
            in_c = out_c
        self.features = nn.Sequential(*layers)

        # --------- 任务分支 --------- #
        if task == "classification":
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_c, num_classes),
            )
        else:  # regression
            self.head = nn.Sequential(
                nn.Conv2d(in_c, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 3, kernel_size=1),
                nn.Sigmoid(),
            )

        # --------- 参数初始化 --------- #
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (N, 3, 100, 200)
        """
        x = self.features(x)
        x = self.head(x)

        if self.task == "regression":
            # 如果 dilation 组合导致分辨率改变，则双线性插值回原尺寸
            if x.shape[-2:] != (self.IMG_H, self.IMG_W):
                x = nn.functional.interpolate(
                    x, size=(self.IMG_H, self.IMG_W), mode="bilinear", align_corners=False
                )
        return x
