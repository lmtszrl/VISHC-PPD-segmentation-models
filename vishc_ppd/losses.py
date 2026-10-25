"""Compound segmentation loss: cross-entropy + λ · soft Dice."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import NUM_CLASSES

__all__ = ["SoftDiceLoss", "CompoundLoss"]


class SoftDiceLoss(nn.Module):
    """
    Multi-class soft Dice loss computed per class over the whole batch.

    Background (``ignore_index``) is excluded from the Dice term so that the
    small toe regions are not dominated by the large background class.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, ignore_index: int = 0, smooth: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits.float(), dim=1)                     # (N, C, H, W)
        onehot = F.one_hot(target.clamp(0, self.num_classes - 1), self.num_classes)
        onehot = onehot.permute(0, 3, 1, 2).to(probs.dtype)               # (N, C, H, W)
        keep = [c for c in range(self.num_classes) if c != self.ignore_index]
        probs, onehot = probs[:, keep], onehot[:, keep]
        dims = (0, 2, 3)
        inter = (probs * onehot).sum(dims)
        denom = probs.sum(dims) + onehot.sum(dims)
        dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class CompoundLoss(nn.Module):
    """``CE(logits, y) + dice_weight * SoftDice(logits, y)``."""

    def __init__(self, num_classes: int = NUM_CLASSES, ignore_index: int = 0, dice_weight: float = 1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = SoftDiceLoss(num_classes, ignore_index=ignore_index)
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.ce(logits.float(), target)
        if self.dice_weight > 0:
            loss = loss + self.dice_weight * self.dice(logits, target)
        return loss
