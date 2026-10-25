"""
``PlantarDataset`` and the augmentation / preprocessing pipelines.

Conventions
-----------
* Images are resized to ``size_hw`` (default 512 × 256) and ImageNet-normalised.
* Masks are resized with nearest-neighbour interpolation.
* **No horizontal / vertical flips** — they would swap medial ↔ lateral
  regions (P3 ↔ P5, P6 ↔ P7), which are anatomically distinct classes.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from ..constants import DEFAULT_INPUT_HW, IMAGENET_MEAN, IMAGENET_STD
from .annotations import load_mask
from .splits import Sample

__all__ = ["PlantarDataset", "build_eval_transform", "build_train_transform", "read_image_rgb"]


def read_image_rgb(path: Path | str) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _resize(h: int, w: int) -> A.Resize:
    return A.Resize(h, w, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST)


def build_eval_transform(size_hw: Tuple[int, int] = DEFAULT_INPUT_HW) -> A.Compose:
    h, w = size_hw
    return A.Compose([_resize(h, w), A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])


def build_train_transform(size_hw: Tuple[int, int] = DEFAULT_INPUT_HW) -> A.Compose:
    """
    Geometry: small shift / scale / rotation (no flips).
    Photometric: brightness-contrast jitter, light Gaussian noise, rare blur.
    """
    h, w = size_hw
    return A.Compose([
        _resize(h, w),
        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=(0.90, 1.10),
            rotate=(-10, 10),
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
            p=0.70,
        ),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.40),
        A.GaussNoise(std_range=(0.02, 0.06), p=0.15),
        A.Blur(blur_limit=3, p=0.10),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class PlantarDataset(Dataset):
    """
    Yields ``(image_tensor[3,H,W] float, mask_tensor[H,W] long, image_id)``.

    Samples without an annotation yield an all-background mask (useful for
    inference-only folders).
    """

    def __init__(
        self,
        samples: Sequence[Sample],
        size_hw: Tuple[int, int] = DEFAULT_INPUT_HW,
        train: bool = False,
    ):
        self.samples: List[Sample] = list(samples)
        self.size_hw = tuple(size_hw)
        self.train = train
        self.transform = build_train_transform(self.size_hw) if train else build_eval_transform(self.size_hw)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = read_image_rgb(s.image_path)
        h0, w0 = img.shape[:2]
        if s.has_annotation:
            mask = load_mask(s.json_path, h0, w0)
        else:
            mask = np.zeros((h0, w0), dtype=np.uint8)
        out = self.transform(image=img, mask=mask)
        x = out["image"].float()
        y = torch.as_tensor(np.asarray(out["mask"]), dtype=torch.long)
        return x, y, s.image_id
