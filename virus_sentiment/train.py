#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fine-tune Chinese RoBERTa on 6-way Weibo emotion classification.
兼容 transformers ≥ 4.46（使用 eval_strategy）。
"""

from __future__ import annotations
import argparse
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from dataset import WeiboDataset, LABEL2ID, ID2LABEL
from evaluate import load as load_metric


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Weibo emotion fine-tuning")
    p.add_argument("--train_file", required=True)
    p.add_argument("--eval_file", required=True)
    p.add_argument("--model_name", default="hfl/chinese-roberta-wwm-ext")
    p.add_argument("--output_dir", default="checkpoint")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--no_cuda", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_ds = WeiboDataset(args.train_file, tokenizer)
    eval_ds = WeiboDataset(args.eval_file, tokenizer)

    metric_acc = load_metric("accuracy")
    metric_f1 = load_metric("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": metric_acc.compute(predictions=preds, references=labels)["accuracy"],
            "f1_macro": metric_f1.compute(predictions=preds, references=labels,
                                          average="macro")["f1"],
        }

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",  # transformers ≥4.46
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        no_cuda=args.no_cuda,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"✅ 训练完成，模型已保存到 {args.output_dir}")


if __name__ == "__main__":
    main()
