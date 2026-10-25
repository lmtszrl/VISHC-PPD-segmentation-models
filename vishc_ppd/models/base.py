"""Common interface for all benchmark models."""

from __future__ import annotations

from typing import Dict, Iterator

import torch
import torch.nn as nn


class SegmentationModel(nn.Module):
    """
    Every benchmark model returns ``{"out": logits}`` with logits of shape
    ``(N, num_classes, H, W)`` at the input resolution, plus an optional
    ``"aux"`` entry used only during training (DeepLabV3).

    ``backbone_parameters()`` / ``head_parameters()`` expose the two parameter
    groups used for the freeze-then-fine-tune schedule and for layer-wise
    learning rates.
    """

    #: human-readable name used in tables and figures
    display_name: str = "model"

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:  # pragma: no cover - abstract
        raise NotImplementedError

    def backbone_parameters(self) -> Iterator[nn.Parameter]:  # pragma: no cover - abstract
        raise NotImplementedError

    def head_parameters(self) -> Iterator[nn.Parameter]:
        backbone_ids = {id(p) for p in self.backbone_parameters()}
        return (p for p in self.parameters() if id(p) not in backbone_ids)

    def count_parameters(self, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in self.parameters() if (p.requires_grad or not trainable_only))
