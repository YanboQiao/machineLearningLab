"""
timer.py
========
简单的计时上下文管理器，用法：

>>> with Timer("epoch"):
>>>     train(...)
"""

from __future__ import annotations

import time
from typing import Optional


class Timer:
    def __init__(self, name: Optional[str] = None, verbose: bool = False):
        self.name = name
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.time() - self.start
        if self.verbose and self.name:
            print(f"[Timer] {self.name}: {self.elapsed:.3f}s")