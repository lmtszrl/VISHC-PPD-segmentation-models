"""
SegFormer-B2 (MiT-B2 encoder + all-MLP decoder).

Xie et al., "SegFormer: Simple and Efficient Design for Semantic Segmentation
with Transformers", NeurIPS 2021.  The implementation is the one shipped in
the ``transformers`` library; the encoder is initialised from the official
ImageNet-1k checkpoint ``nvidia/mit-b2`` (downloaded once, no account
required) and the decode head is trained from scratch.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from ..constants import NUM_CLASSES
from .base import SegmentationModel

HF_ENCODER = "nvidia/mit-b2"


def _config(num_classes: int):
    from transformers import SegformerConfig

    return SegformerConfig(
        num_channels=3,
        depths=[3, 4, 6, 3],
        hidden_sizes=[64, 128, 320, 512],
        decoder_hidden_size=768,
        num_attention_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        num_labels=num_classes,
    )


class SegFormerB2(SegmentationModel):
    display_name = "SegFormer-B2"

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True, **_):
        super().__init__()
        from transformers import SegformerForSemanticSegmentation

        cfg = _config(num_classes)
        if pretrained:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                HF_ENCODER, config=cfg, ignore_mismatched_sizes=True
            )
        else:
            self.model = SegformerForSemanticSegmentation(cfg)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.model(pixel_values=x).logits            # (N, C, H/4, W/4)
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return {"out": logits}

    def backbone_parameters(self):
        return self.model.segformer.parameters()
