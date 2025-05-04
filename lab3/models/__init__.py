"""
models 包

统一暴露四个网络：

    • HandConvNet —— “手写二维卷积”版本（不依赖 torch.nn.Conv2d）
    • SimpleCNN   —— 使用 torch.nn.Conv2d 的基础 CNN，可调层数 / kernel_size
    • AlexNet     —— 改写自经典 AlexNet，使其能接受 (3,100,200) 输入
    • FFNN        —— 纯前馈全连接网络（Flatten → Linear*）

>>> from models import HandConvNet, SimpleCNN, AlexNet, FFNN
"""
from .hand_conv import HandConvNet
from .cnn_torch import SimpleCNN
from .alexnet import AlexNet
from .ffnn import FFNN
from .dilated_cnn import DilatedCNN

__all__ = ["HandConvNet", "SimpleCNN", "AlexNet", "FFNN", "DilatedCNN"]