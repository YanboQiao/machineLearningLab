"""
eval.py
=======
单次推理 / 评估脚本

• 计算 loss 与 metric
• （可选）在回归任务中将预测结果保存为可视化图片
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from utils.metrics import accuracy, psnr
except Exception:
    import torch.nn.functional as F

    def accuracy(logits, target):
        pred = logits.argmax(dim=1)
        return (pred == target).float().mean().item()

    def psnr(pred, target):
        mse = F.mse_loss(pred, target, reduction="mean").item()
        return float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse)


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion,
    *,
    device: str | torch.device | None = None,
    task: str = "classification",
    save_samples: bool = False,
    samples_dir: str | Path = "samples",
    num_samples: int = 6,
) -> Dict[str, float]:
    """
    Parameters
    ----------
    model : 已训练模型
    loader : 测试集 DataLoader
    criterion : 损失函数
    device : 推理设备
    task : "classification" | "regression"
    save_samples : 若 True 并且是去雾任务，则保存 n 张 按 “原图-预测-真值” 拼接图
    samples_dir : 图片输出目录
    num_samples : 保存多少张示例
    """
    assert task in {"classification", "regression"}

    dvc = torch.device(
        device
        or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    )
    model.to(dvc).eval()

    metric_fn = accuracy if task == "classification" else psnr

    running_loss, running_metric, n_batches = 0.0, 0.0, 0
    collected = 0
    samples_dir = Path(samples_dir)
    if save_samples and task == "regression":
        samples_dir.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(loader, desc="[Eval]", leave=False)
    for x, y in pbar:
        x, y = x.to(dvc), y.to(dvc)
        out = model(x)
        loss = criterion(out, y)

        running_loss += loss.item()
        running_metric += metric_fn(out, y)
        n_batches += 1

        pbar.set_postfix({"loss": running_loss / n_batches, "metric": running_metric / n_batches})

        # --- 保存示例 --- #
        if save_samples and task == "regression" and collected < num_samples:
            # 反归一化到 [0,255] uint8
            for i in range(min(x.size(0), num_samples - collected)):
                fog = (x[i].detach().cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255).astype("uint8")
                pred = (out[i].detach().cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255).astype("uint8")
                gt = (y[i].detach().cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255).astype("uint8")

                # 横向拼接
                concat = Image.fromarray(
                    torch.from_numpy(
                        torch.cat(
                            [
                                torch.from_numpy(fog),
                                torch.from_numpy(pred),
                                torch.from_numpy(gt),
                            ],
                            dim=1,
                        ).numpy()
                    ).numpy()
                )
                concat.save(samples_dir / f"sample_{collected:02d}.png")
                collected += 1
                if collected >= num_samples:
                    break

    return {
        "loss": running_loss / n_batches,
        "metric": running_metric / n_batches,
    }