"""
engine 包

主要公开两大接口：
    • train     —— 完整的训练流程（含验证、学习率调度与最佳模型保存）
    • evaluate  —— 单次评估 / 推理

>>> from engine import train, eval
"""
from .train import train
from .eval import evaluate

__all__ = ["train", "evaluate"]