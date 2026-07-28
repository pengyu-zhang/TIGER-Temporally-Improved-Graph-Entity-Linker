"""Shared utilities: seeding, device selection, logging, JSONL metrics."""

import json
import logging
import os
import random
import sys
import time

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fix every random source we use."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(no_cuda: bool = False) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        print(f"[tiger] device: {device} ({name})", flush=True)
    else:
        print(f"[tiger] device: {device}", flush=True)
    return device


def configure_backends(tf32: bool) -> None:
    """TF32 switch (Ampere+). No-op on CPU."""
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32


def get_logger(output_dir: str | None = None, name: str = "tiger") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%m/%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(output_dir, "log.txt"), mode="a")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


class JsonlWriter:
    """Append-only JSONL metrics writer; every line is flushed so an
    interrupted run keeps everything written so far."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")

    def write(self, record: dict) -> None:
        record = {"time": round(time.time(), 3), **record}
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def read_jsonl(path: str, limit: int | None = None) -> list[dict]:
    samples = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
            if limit is not None and len(samples) >= limit:
                break
    return samples
