#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估 fine-tuned 模型在验证集上的准确率 / 宏 F1。
"""

import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, f1_score
from dataset import WeiboDataset, ID2LABEL


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Weibo emotion model")
    p.add_argument("--model_dir", required=True)
    p.add_argument("--eval_file", required=True)
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)

    ds = WeiboDataset(args.eval_file, tokenizer)
    loader = DataLoader(ds, batch_size=args.batch_size)

    preds, labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            labels.extend(batch["labels"].tolist())
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            ).logits
            preds.extend(np.argmax(logits.cpu().numpy(), axis=-1))

    print("Accuracy:", accuracy_score(labels, preds))
    print("Macro-F1:", f1_score(labels, preds, average="macro"))
    print(classification_report(labels, preds,
                                target_names=[ID2LABEL[i] for i in range(6)]))


if __name__ == "__main__":
    main()
