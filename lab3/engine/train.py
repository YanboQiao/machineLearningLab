"""
train.py
========
通用 PyTorch 训练循环，支持
    • 三分类任务（accuracy）
    • 去雾回归任务（PSNR）

已更新为 **torch.amp** 新 API，并在非-CUDA 设备上自动关闭 AMP，
从而消除 FutureWarning / UserWarning。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

# ------------------------------------------------------------ #
# 指标函数 —— 如果 utils.metrics 尚未实现，用兜底版本
# ------------------------------------------------------------ #
try:
    from utils.metrics import accuracy, psnr
except Exception:
    import torch.nn.functional as F
    import math

    def accuracy(logits, target):
        pred = logits.argmax(dim=1)
        return (pred == target).float().mean().item()

    def psnr(pred, target):
        mse = F.mse_loss(pred, target, reduction="mean").item()
        return float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse)


# ------------------------------------------------------------ #
# 单个 epoch 训练
# ------------------------------------------------------------ #
def _train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
    scaler: Optional[GradScaler],
    task: str,
    use_amp: bool,
) -> Dict[str, float]:
    """
    返回 {"loss": …, "metric": …}
    """
    model.train()
    metric_fn = accuracy if task == "classification" else psnr

    running_loss = running_metric = 0.0
    n_batches = 0
    pbar = tqdm(loader, desc="[Train]", leave=False)

    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            # 新 API 需显式声明 device_type='cuda'
            with autocast(device_type="cuda", enabled=True):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        # 统计
        running_loss += loss.item()
        running_metric += metric_fn(out.detach(), y.detach())
        n_batches += 1
        pbar.set_postfix(
            {"loss": running_loss / n_batches, "metric": running_metric / n_batches}
        )

    return {"loss": running_loss / n_batches, "metric": running_metric / n_batches}


# ------------------------------------------------------------ #
# 验证
# ------------------------------------------------------------ #
@torch.inference_mode()
def _validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
    task: str,
) -> Dict[str, float]:
    model.eval()
    metric_fn = accuracy if task == "classification" else psnr

    running_loss = running_metric = 0.0
    n_batches = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        running_loss += loss.item()
        running_metric += metric_fn(out, y)
        n_batches += 1

    return {"loss": running_loss / n_batches, "metric": running_metric / n_batches}


# ------------------------------------------------------------ #
# 训练主入口
# ------------------------------------------------------------ #
def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion,
    optimizer,
    *,
    device: str | torch.device | None = None,
    num_epochs: int = 20,
    task: str = "classification",
    scheduler=None,
    amp: bool = True,
    save_best: bool = False,
    save_path: str | Path = "best_model.pt",
    verbose: bool = True,
) -> Dict[str, list]:
    """
    通用训练接口，返回可直接馈入 utils.visualizer.plot_history 的字典
    """
    assert task in {"classification", "regression"}

    # -------- 设备选择 -------- #
    dvc = torch.device(
        device
        or (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    )
    model.to(dvc)

    # -------- AMP 设置 -------- #
    use_amp = amp and dvc.type == "cuda"
    scaler = GradScaler(device_type="cuda") if use_amp else None

    # -------- 历史记录 -------- #
    history = {k: [] for k in ("epoch", "train_loss", "val_loss", "train_metric", "val_metric", "epoch_time")}

    best_val_metric = -float("inf") if task == "classification" else float("inf")
    better = (lambda a, b: a > b) if task == "classification" else (lambda a, b: a < b)

    if verbose:
        amp_state = "enabled" if use_amp else "disabled"
        print(f"Start Training on {dvc} ({task}, AMP {amp_state}) for {num_epochs} epochs.")

    # ==================== 主循环 ==================== #
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_stats = _train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            dvc,
            scaler,
            task,
            use_amp=use_amp,
        )
        val_stats = _validate(model, val_loader, criterion, dvc, task)

        if scheduler is not None:
            scheduler.step(val_stats["loss"])

        epoch_time = time.time() - t0

        # ---- 记录 ---- #
        history["epoch"].append(epoch)
        history["train_loss"].append(train_stats["loss"])
        history["val_loss"].append(val_stats["loss"])
        history["train_metric"].append(train_stats["metric"])
        history["val_metric"].append(val_stats["metric"])
        history["epoch_time"].append(epoch_time)

        if verbose:
            print(
                f"[{epoch:02d}/{num_epochs}] "
                f"Train Loss: {train_stats['loss']:.4f} | "
                f"Val Loss: {val_stats['loss']:.4f} | "
                f"Train Metric: {train_stats['metric']:.4f} | "
                f"Val Metric: {val_stats['metric']:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )

        # ---- 保存最佳模型 ---- #
        if save_best and better(val_stats["metric"], best_val_metric):
            best_val_metric = val_stats["metric"]
            torch.save(model.state_dict(), str(save_path))
            if verbose:
                print(f"  ↳ New best model saved to {save_path} (metric={best_val_metric:.4f})")

    return history
