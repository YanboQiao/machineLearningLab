#!/usr/bin/env python
"""
run_vehicle.py
==============
车辆三分类实验入口脚本

用法（示例）::
    # 直接使用默认超参数
    python run_vehicle.py

    # 指定 YAML 配置文件覆盖默认值
    python run_vehicle.py --config configs/vehicle.yaml

    # 命令行临时覆盖某些参数
    python run_vehicle.py --epochs 30 --model alexnet --batch_size 64
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

from data import VehicleDataset, build_dataloaders
from engine import evaluate, train
from models import AlexNet, HandConvNet, SimpleCNN
from utils.visualizer import plot_history


# ----------------------- 默认超参数 ----------------------- #
DEFAULTS: Dict[str, Any] = dict(
    dataset_root="Datasets/viecleClassificationDataset",
    model="simple_cnn",           # {"simple_cnn", "hand_conv", "alexnet"}
    epochs=20,
    batch_size=32,
    lr=1e-3,
    conv_channels=[32, 64, 128],  # 对 simple_cnn/hand_conv 生效
    save_best=True,
    out_dir="runs/vehicle",
)


# ----------------------- CLI & 配置 ----------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vehicle Classification Trainer")
    parser.add_argument("--config", type=str, default=None, help="YAML/JSON config file")
    parser.add_argument("--dataset_root", type=str, help="Dataset root directory")
    parser.add_argument("--model", type=str, choices=["simple_cnn", "hand_conv", "alexnet"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--conv_channels", type=str, help="e.g. 32,64,128")
    parser.add_argument("--out_dir", type=str)
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
    # conv_channels: "32,64,128" -> [32,64,128]
    if "conv_channels" in cli_dict and isinstance(cli_dict["conv_channels"], str):
        cli_dict["conv_channels"] = [int(x) for x in cli_dict["conv_channels"].split(",")]
    merged.update(cli_dict)
    return merged


# ----------------------- 主流程 ----------------------- #
def get_model(name: str, params: Dict[str, Any]) -> torch.nn.Module:
    if name == "simple_cnn":
        return SimpleCNN(task="classification", conv_channels=params["conv_channels"])
    if name == "hand_conv":
        return HandConvNet(task="classification", conv_channels=params["conv_channels"])
    if name == "alexnet":
        return AlexNet(task="classification")
    raise ValueError(f"Unknown model {name}")


def main():
    cli = parse_args()
    cfg_file = load_config(cli.config)
    params = merge_args(cfg_file, cli)

    out_dir = Path(params["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------- Dataset & DataLoader -------- #
    train_loader, val_loader = build_dataloaders(
        VehicleDataset,
        root=params["dataset_root"],
        batch_size=params["batch_size"],
        num_workers=4,
    )

    # -------- Model / Loss / Optim -------- #
    model = get_model(params["model"], params)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

    # -------- Train -------- #
    history = train(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=params["epochs"],
        task="classification",
        save_best=params["save_best"],
        save_path=out_dir / "best_model.pt",
    )

    plot_history(history, save_path=out_dir / "history.png")

    # -------- Evaluate Best -------- #
    if params["save_best"]:
        model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location="cpu"))
    test_stats = evaluate(
        model,
        val_loader,
        criterion,
        task="classification",
    )
    print(f"\n[Final] Accuracy={test_stats['metric']:.4f}  Loss={test_stats['loss']:.4f}")


if __name__ == "__main__":
    main()
