#!/usr/bin/env python
"""
run_defog.py
============
图片去雾回归实验入口

示例::
    python run_defog.py --model hand_conv --epochs 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import yaml

from data import DefogDataset, build_dataloaders
from engine import evaluate, train
from models import AlexNet, FFNN, HandConvNet, SimpleCNN
from utils.visualizer import plot_history


# ----------------------- 默认超参数 ----------------------- #
DEFAULTS: Dict[str, Any] = dict(
    dataset_root="Datasets/defogDataset",
    model="simple_cnn",         # {"simple_cnn", "hand_conv", "alexnet", "ffnn"}
    epochs=25,
    batch_size=16,
    lr=1e-3,
    conv_channels=[32, 64, 128],
    save_best=True,
    out_dir="runs/defog",
    save_samples=True,
    samples_num=6,
)


# ----------------------- CLI & 配置 ----------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Image Defogging Trainer")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dataset_root", type=str)
    parser.add_argument("--model", type=str, choices=["simple_cnn", "hand_conv", "alexnet", "ffnn"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--conv_channels", type=str)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--samples_num", type=int)
    return parser.parse_args()


def load_config(path: str | None) -> Dict[str, Any]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        sys.exit(f"[Error] Config file {path} not found.")
    with open(path, "r", encoding="utf-8") as f:
        if path.suffix in {".yml", ".yaml"}:
            return yaml.safe_load(f)
        elif path.suffix == ".json":
            return json.load(f)
        else:
            sys.exit("Config file must be .yaml, .yml, or .json")


def merge_args(cfg: Dict[str, Any], cli: argparse.Namespace) -> Dict[str, Any]:
    merged = DEFAULTS.copy()
    merged.update(cfg)
    cli_dict = {k: v for k, v in vars(cli).items() if v is not None}
    if "conv_channels" in cli_dict and isinstance(cli_dict["conv_channels"], str):
        cli_dict["conv_channels"] = [int(x) for x in cli_dict["conv_channels"].split(",")]
    merged.update(cli_dict)
    return merged


# ----------------------- 主流程 ----------------------- #
def get_model(name: str, params: Dict[str, Any]) -> torch.nn.Module:
    if name == "simple_cnn":
        return SimpleCNN(task="regression", conv_channels=params["conv_channels"])
    if name == "hand_conv":
        return HandConvNet(task="regression", conv_channels=params["conv_channels"])
    if name == "alexnet":
        return AlexNet(task="regression")
    if name == "ffnn":
        return FFNN(task="regression")
    raise ValueError(f"Unknown model {name}")


def main():
    cli = parse_args()
    cfg_file = load_config(cli.config)
    params = merge_args(cfg_file, cli)

    out_dir = Path(params["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------- Dataset -------- #
    train_loader, val_loader = build_dataloaders(
        DefogDataset,
        root=params["dataset_root"],
        batch_size=params["batch_size"],
        num_workers=4,
    )

    # -------- Model & Optim -------- #
    model = get_model(params["model"], params)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

    # -------- Train -------- #
    history = train(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=params["epochs"],
        task="regression",
        save_best=params["save_best"],
        save_path=out_dir / "best_model.pt",
    )

    plot_history(history, save_path=out_dir / "history.png")

    # -------- Evaluate -------- #
    if params["save_best"]:
        model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location="cpu"))

    test_stats = evaluate(
        model,
        val_loader,
        criterion,
        task="regression",
        save_samples=params["save_samples"],
        samples_dir=out_dir / "samples",
        num_samples=params["samples_num"],
    )
    print(f"\n[Final] PSNR={test_stats['metric']:.2f}dB  Loss={test_stats['loss']:.6f}")


if __name__ == "__main__":
    main()