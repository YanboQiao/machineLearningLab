from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple
from torch.utils.data import Dataset

LABEL2ID = {"happy": 0, "angry": 1, "sad": 2,
            "fear": 3, "surprise": 4, "neural": 5}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class WeiboDataset(Dataset):
    """
    支持两种数据格式：
      1. JSON-Lines：一行一个 {"content": ..., "label": ...}
      2. JSON 数组：整个文件是 [ {...}, {...}, ... ]
    """

    def __init__(self,
                 path: str | Path,
                 tokenizer,
                 max_length: int = 128) -> None:
        path = Path(path)
        raw = path.read_text(encoding="utf-8").strip()

        # 判断文件是数组还是 JSON-Lines
        records = (json.loads(raw) if raw.startswith("[")
                   else [json.loads(l) for l in raw.splitlines() if l.strip()])

        self.samples: List[Tuple[str, int]] = [
            (r["content"], LABEL2ID.get(r.get("label", "neural"), LABEL2ID["neural"]))
            for r in records
        ]
        self.tokenizer = tokenizer
        self.max_length = max_length

    # -------------- Dataset 接口 --------------
    def __len__(self) -> int:  # noqa: D401
        return len(self.samples)

    def __getitem__(self, idx: int):
        text, label = self.samples[idx]
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = label
        return item
