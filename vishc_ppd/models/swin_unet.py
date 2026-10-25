"""
Swin-UNet: Swin-Tiny hierarchical encoder (timm) in a U-Net topology with a
convolutional decoder.

Cao et al., "Swin-Unet: Unet-like Pure Transformer for Medical Image
Segmentation", ECCV 2022 Workshops.  The original Swin-Tiny encoder is kept;
the decoder uses convolutional up-blocks, which train more stably than a
patch-expanding decoder on a dataset of this size.

Note: the timm Swin encoder is built for a fixed input size, so ``input_hw``
must equal the training / inference resolution (stored in the checkpoint).
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from ..constants import DEFAULT_INPUT_HW, NUM_CLASSES
from .base import SegmentationModel
from .unet import UpBlock

TIMM_ENCODER = "swin_tiny_patch4_window7_224"


class SwinUNet(SegmentationModel):
    display_name = "Swin-UNet (T)"

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True,
                 input_hw: Tuple[int, int] = DEFAULT_INPUT_HW, **_):
        super().__init__()
        import timm

        self.encoder = timm.create_model(
            TIMM_ENCODER, pretrained=pretrained, features_only=True,
            out_indices=(0, 1, 2, 3), img_size=tuple(input_hw),
            # official Swin release (GitHub) rather than the default hub mirror
            pretrained_cfg_overlay=dict(hf_hub_id=None, hf_hub_filename=None),
        )
        ch = self.encoder.feature_info.channels()   # [96, 192, 384, 768]
        self.up3 = UpBlock(ch[3], ch[2], 256)
        self.up2 = UpBlock(256, ch[1], 128)
        self.up1 = UpBlock(128, ch[0], 64)
        self.up0 = UpBlock(64, 0, 32)
        self.up_final = UpBlock(32, 0, 16)
        self.head = torch.nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.encoder(x)
        # timm returns NHWC feature maps for Swin; convert to NCHW
        feats = [f.permute(0, 3, 1, 2).contiguous() if f.shape[-1] == c and f.shape[1] != c else f
                 for f, c in zip(feats, self.encoder.feature_info.channels())]
        f0, f1, f2, f3 = feats
        d = self.up3(f3, f2)
        d = self.up2(d, f1)
        d = self.up1(d, f0)
        d = self.up0(d)
        d = self.up_final(d)
        logits = self.head(d)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return {"out": logits}

    def backbone_parameters(self):
        return self.encoder.parameters()
