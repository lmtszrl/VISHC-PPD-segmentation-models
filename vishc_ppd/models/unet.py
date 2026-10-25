"""
U-Net with an ImageNet-pretrained ResNet-34 encoder.

Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image
Segmentation", MICCAI 2015.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet34_Weights, resnet34

from ..constants import NUM_CLASSES
from .base import SegmentationModel


class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))


class UpBlock(nn.Module):
    """Bilinear ×2 up-sampling → concat skip → two ConvBnRelu layers."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(ConvBnRelu(in_ch + skip_ch, out_ch), ConvBnRelu(out_ch, out_ch))

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        else:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv(x)


class UNet(SegmentationModel):
    display_name = "U-Net (R34)"

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True, **_):
        super().__init__()
        bb = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
        self.enc0 = nn.Sequential(bb.conv1, bb.bn1, bb.relu)   # /2,  64
        self.pool = bb.maxpool
        self.enc1, self.enc2, self.enc3, self.enc4 = bb.layer1, bb.layer2, bb.layer3, bb.layer4  # /4 64, /8 128, /16 256, /32 512
        self.up4 = UpBlock(512, 256, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up1 = UpBlock(64, 64, 64)
        self.up0 = UpBlock(64, 0, 32)
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        d = self.up4(e4, e3)
        d = self.up3(d, e2)
        d = self.up2(d, e1)
        d = self.up1(d, e0)
        d = self.up0(d)
        logits = self.head(d)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return {"out": logits}

    def backbone_parameters(self):
        return (p for m in (self.enc0, self.enc1, self.enc2, self.enc3, self.enc4) for p in m.parameters())
