"""
utils.py
--------
Shared helpers for train / test scripts.
"""

import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------
def get_logger(log_dir: str, name: str = "spectra_avqa") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = Path(log_dir) / f"{name}_{stamp}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            "%H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(f"Log file: {log_path}")
    return logger


# ---------------------------------------------------------------------
class AverageMeter:
    def __init__(self):
        self.n = 0; self.sum = 0.0
    def update(self, val: float, k: int = 1):
        self.sum += val * k; self.n += k
    @property
    def avg(self) -> float:
        return self.sum / max(self.n, 1)


# ---------------------------------------------------------------------
def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return (pred == labels).float().mean().item()


# ---------------------------------------------------------------------
class PerTypeAccuracy:
    """Accumulate accuracy broken down by `q_type` string."""

    def __init__(self):
        self.correct = defaultdict(int)
        self.total = defaultdict(int)

    def update(self, logits: torch.Tensor, labels: torch.Tensor, types: list[str]):
        pred = logits.argmax(dim=-1).cpu()
        labels = labels.cpu()
        for p, l, t in zip(pred.tolist(), labels.tolist(), types):
            key = t if t else "All"
            self.total[key] += 1
            if p == l:
                self.correct[key] += 1

    def report(self) -> dict[str, float]:
        out = {k: self.correct[k] / max(self.total[k], 1) for k in self.total}
        tot_c = sum(self.correct.values()); tot_n = sum(self.total.values())
        out["overall"] = tot_c / max(tot_n, 1)
        return out


# ---------------------------------------------------------------------
def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
