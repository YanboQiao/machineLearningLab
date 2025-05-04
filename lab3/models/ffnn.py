"""
ffnn.py
=======
前馈全连接网络：输入 Flatten → Linear × N
"""

from __future__ import annotations

from typing import List

import torch
from torch import nn


class FFNN(nn.Module):
    """
    Parameters
    ----------
    task : "classification" | "regression"
    num_classes : 分类时类别数
    hidden_sizes : List[int] —— 隐藏层神经元数
    """

    IMG_SHAPE = (3, 100, 200)
    FLAT_DIM = IMG_SHAPE[0] * IMG_SHAPE[1] * IMG_SHAPE[2]

    def __init__(
        self,
        *,
        task: str = "classification",
        num_classes: int = 3,
        hidden_sizes: List[int] | None = None,
    ):
        super().__init__()
        assert task in {"classification", "regression"}
        self.task = task

        hidden_sizes = hidden_sizes or [512, 256, 128]

        sizes = [self.FLAT_DIM] + hidden_sizes
        layers = []
        for in_f, out_f in zip(sizes[:-1], sizes[1:]):
            layers += [nn.Linear(in_f, out_f), nn.ReLU(inplace=True)]
        if task == "classification":
            layers.append(nn.Linear(sizes[-1], num_classes))
        else:
            layers.append(nn.Linear(sizes[-1], self.FLAT_DIM))
            layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        x = self.net(x)
        if self.task == "regression":
            # reshape back to (N, 3, 100, 200)
            x = x.view(-1, *FFNN.IMG_SHAPE)
        return x