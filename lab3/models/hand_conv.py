"""
hand_conv.py
============
“手写二维卷积”实现：完全不调用 torch.nn.Conv2d，而是用
torch.nn.functional.unfold + torch.matmul 自行完成卷积运算，
同时保持梯度可回传，方便直接与 PyTorch Optimizer/Autograd
协同训练。

本文件暴露两个类
    • MyConv2d   —— 单层卷积算子（可作替代 Conv2d）
    • HandConvNet —— 使用 MyConv2d 叠若干层得到的网络
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class MyConv2d(nn.Module):
    """
    纯手写卷积层 —— Parameter + unfold + matmul

    Parameters
    ----------
    in_channels, out_channels : 输入/输出通道
    kernel_size : 单个 int 或 (kh, kw)
    stride, padding, bias     : 与 nn.Conv2d 同语义
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int],
        stride: int | Tuple[int, int] = 1,
        padding: int | Tuple[int, int] = 0,
        bias: bool = True,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        weight_shape = (out_channels, in_channels, *kernel_size)
        self.weight = nn.Parameter(torch.empty(weight_shape))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        # Kaiming 正态初始化
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        x : (N, C_in, H, W)

        输出： (N, C_out, H_out, W_out)
        """
        N, C, H, W = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding

        # unfold : (N, C*kh*kw, L) where L = H_out * W_out
        x_unf = F.unfold(x, kernel_size=(kh, kw), stride=(sh, sw), padding=(ph, pw))

        # weight_flat : (C_out, C_in*kh*kw)
        w_flat = self.weight.view(self.out_channels, -1)

        # y_flat : (N, C_out, L)
        y_flat = w_flat @ x_unf
        if self.bias is not None:
            y_flat += self.bias[:, None]

        # 重塑回 N,C_out,H_out,W_out
        H_out = (H + 2 * ph - kh) // sh + 1
        W_out = (W + 2 * pw - kw) // sw + 1
        y = y_flat.view(N, self.out_channels, H_out, W_out)
        return y


class HandConvNet(nn.Module):
    """
    由 *手写* 卷积层堆叠而成的 CNN

    适配两种任务：
        • 分类         —— task='classification'，输出 shape=(N, num_classes)
        • 去雾回归     —— task='regression'，输出 shape=(N, 3, 100, 200)

    超参数
    -------
    conv_channels : List[int]   —— 每层卷积输出通道数
    kernel_size   : int         —— 所有卷积使用统一 kernel_size
    """

    def __init__(
        self,
        *,
        task: str = "classification",
        num_classes: int = 3,
        conv_channels: List[int] | None = None,
        kernel_size: int = 3,
    ):
        super().__init__()
        assert task in {"classification", "regression"}
        self.task = task

        conv_channels = conv_channels or [16, 32, 64]

        layers: List[nn.Module] = []
        in_c = 3
        for out_c in conv_channels:
            layers.append(
                MyConv2d(
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,  # same padding
                )
            )
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=2))
            in_c = out_c

        self.features = nn.Sequential(*layers)

        if task == "classification":
            # 全局平均池得到 (N, C_last)
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_c, num_classes),
            )
        else:  # regression
            # 反卷积 / 上采样还原到 (3,100,200)
            self.reg_head = nn.Sequential(
                nn.ConvTranspose2d(in_c, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 3, kernel_size=1),
                nn.Sigmoid(),  # 归一化输出 0~1
            )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        if self.task == "classification":
            return self.classifier(x)
        return self.reg_head(x)