"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

__all__ = ["seed_everything"]


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch; optionally force deterministic cuDNN."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
