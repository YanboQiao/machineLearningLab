#!/usr/bin/env python3
"""
run_dilated.py
==============
在去雾数据集上比较
    • DilatedCNN   —— 多层空洞卷积 (满足 HDC)
    • SimpleCNN    —— 普通 3×3 卷积 baseline

功能：
    • 记录总训练时长
    • 保存 loss / PSNR 曲线 (history.png)
    • 保存示例拼接图到 samples/
    • 将结果追加到 runs/results.csv

⚠ 重要：脚本主体放在 `main()`，并用
        if __name__ == "__main__": main()
      保护，避免 macOS / Windows 的
      spawn 多进程 DataLoader 递归闪退。
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from data import DefogDataset, build_dataloaders
from engine import evaluate, train
from models import DilatedCNN, SimpleCNN
from utils.visualizer import plot_history


# ------------------------------ 入口函数 ------------------------------ #
def main():
    # -------------------------- CLI -------------------------- #
    parser = argparse.ArgumentParser("Dilated CNN Defogging Experiment")
    parser.add_argument("--model", choices=["dilated", "baseline"], default="dilated")
    parser.add_argument("--dilations", type=str, default="1,2,5",
                        help="Comma-separated dilation list for DilatedCNN")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers; set 0 for debug")
    parser.add_argument("--dataset_root", type=str, default="Datasets/defogDataset")
    args = parser.parse_args()

    # ------------------------ 目录 ------------------------ #
    tag = (
        f"dilated_{args.dilations.replace(',', '-')}"
        if args.model == "dilated"
        else "baseline"
    )
    out_dir = Path(f"runs/defog_{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------ Data ------------------------ #
    train_loader, val_loader = build_dataloaders(
        DefogDataset,
        root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ----------------------- Model ------------------------ #
    if args.model == "dilated":
        dilation_list = [int(d) for d in args.dilations.split(",")]
        model = DilatedCNN(task="regression", dilation_list=dilation_list)
    else:
        model = SimpleCNN(task="regression")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ----------------------- Train ------------------------ #
    print(f"\n=== Training {tag} ===")
    t0 = time.time()
    history = train(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        task="regression",
        num_epochs=args.epochs,
        save_best=True,
        save_path=out_dir / "best.pt",
    )
    total_time = time.time() - t0
    plot_history(history, out_dir / "history.png")

    # ---------------------- Evaluate ---------------------- #
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location="cpu"))
    test_stats = evaluate(
        model,
        val_loader,
        criterion,
        task="regression",
        save_samples=True,
        samples_dir=out_dir / "samples",
        num_samples=6,
    )

    print(
        f"\n[Finished] {tag}: "
        f"PSNR={test_stats['metric']:.2f} dB | "
        f"Loss={test_stats['loss']:.6f} | "
        f"TrainTime={total_time/60:.1f} min"
    )

    # ----------------------- CSV ------------------------- #
    csv_path = Path("runs/results.csv")
    header = [
        "tag", "model", "dilations", "epochs",
        "batch_size", "lr", "psnr", "loss", "train_time_s"
    ]
    row: Dict[str, str] = {
        "tag": tag,
        "model": args.model,
        "dilations": args.dilations if args.model == "dilated" else "-",
        "epochs": str(args.epochs),
        "batch_size": str(args.batch_size),
        "lr": str(args.lr),
        "psnr": f"{test_stats['metric']:.2f}",
        "loss": f"{test_stats['loss']:.6f}",
        "train_time_s": f"{total_time:.1f}",
    }
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"Results appended to {csv_path.resolve()}")


# --------------- 关键：spawn 兼容保护 ---------------- #
if __name__ == "__main__":
    main()
