"""
metrics.py
==========
分类任务：accuracy
回归任务：MSE、PSNR
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


@torch.inference_mode()
def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    """
    Parameters
    ----------
    logits : (N, C)
    target : (N,)

    Returns
    -------
    acc : float ∈ [0,1]
    """
    pred = logits.argmax(dim=1)
    return (pred == target).float().mean().item()


@torch.inference_mode()
def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    均方误差 (mean over all pixels & channels)
    """
    return F.mse_loss(pred, target, reduction="mean").item()


@torch.inference_mode()
def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    PSNR in dB, assume input ∈ [0,1]
    """
    mse_val = mse_loss(pred, target)
    if mse_val == 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse_val)