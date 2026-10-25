"""
DINOv2 segmentation models: self-supervised ViT encoder + light convolutional decoder.

Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision",
TMLR 2024.

Design
------
* The ViT encoder is created with ``timm`` (the same library that provides the
  Swin-Tiny encoder of Swin-UNet).  Pretrained weights are downloaded once
  from Meta's public release server (``dl.fbaipublicfiles.com``); no account,
  token or third-party model hub is involved.
* Patch tokens of the last four transformer blocks are taken as
  ``(N, C, H/14, W/14)`` feature maps (final LayerNorm applied, as in the
  official ``get_intermediate_layers(norm=True)``), concatenated, projected to
  256 channels and decoded by four ×2 up-sampling Conv-BN-ReLU blocks back to
  the input resolution.
* The patch size is 14.  An input whose side is not a multiple of 14 (the
  benchmark uses 512 × 256) is bilinearly resized to the next multiple before
  the encoder; the decoder returns logits at the original size, so the same
  input protocol applies to every model in the benchmark.  Position
  embeddings are interpolated by timm (``dynamic_img_size=True``).

Model id ``dinov2_vitb14``: timm encoder ``vit_base_patch14_dinov2`` (ViT-B/14,
embedding dim 768, ~89 M parameters in total with the decoder).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..constants import NUM_CLASSES
from .base import SegmentationModel
from .unet import ConvBnRelu


@dataclass(frozen=True)
class DinoVariant:
    timm_name: str
    hidden: int
    patch: int
    display: str


DINO_VARIANTS: Dict[str, DinoVariant] = {
    "dinov2_vitb14": DinoVariant("vit_base_patch14_dinov2", 768, 14, "DINOv2 ViT-B/14"),
}

N_FEATURE_LAYERS = 4      # feature maps taken from the last N transformer blocks


def _build_encoder(v: DinoVariant, pretrained: bool) -> nn.Module:
    import timm

    return timm.create_model(
        v.timm_name,
        pretrained=pretrained,
        num_classes=0,
        dynamic_img_size=True,
        # Take the weights from Meta's release URL recorded in the timm config
        # instead of the default hub mirror.
        pretrained_cfg_overlay=dict(hf_hub_id=None, hf_hub_filename=None),
    )


class DinoDecoder(nn.Module):
    """Concatenated multi-layer features → 1×1 projection → 4 × (×2 up + conv) → logits."""

    def __init__(self, in_ch: int, num_classes: int, width: int = 256):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_ch, width, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        chans = [width, 192, 128, 64, 32]
        self.ups = nn.ModuleList(
            nn.Sequential(ConvBnRelu(chans[i], chans[i + 1]), ConvBnRelu(chans[i + 1], chans[i + 1]))
            for i in range(4)
        )
        self.head = nn.Conv2d(chans[-1], num_classes, 1)

    def forward(self, feat: torch.Tensor, out_hw) -> torch.Tensor:
        x = self.proj(feat)
        for blk in self.ups:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = blk(x)
        logits = self.head(x)
        if logits.shape[-2:] != tuple(out_hw):
            logits = F.interpolate(logits, size=tuple(out_hw), mode="bilinear", align_corners=False)
        return logits


class DinoSeg(SegmentationModel):
    """DINOv2 ViT encoder + convolutional decoder for P1–P8 segmentation."""

    def __init__(self, variant: str = "dinov2_vitb14", num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        if variant not in DINO_VARIANTS:
            raise ValueError(f"unknown DINOv2 variant {variant!r}; choose from {list(DINO_VARIANTS)}")
        self.variant = DINO_VARIANTS[variant]
        self.display_name = self.variant.display
        self.encoder = _build_encoder(self.variant, pretrained)
        self.patch = self.variant.patch
        self.decoder = DinoDecoder(self.variant.hidden * N_FEATURE_LAYERS, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        H, W = x.shape[-2:]
        p = self.patch
        Hp, Wp = -(-H // p) * p, -(-W // p) * p          # next multiples of the patch size
        if (Hp, Wp) != (H, W):
            x = F.interpolate(x, size=(Hp, Wp), mode="bilinear", align_corners=False)
        feats = self.encoder.forward_intermediates(
            x, indices=N_FEATURE_LAYERS, norm=True, output_fmt="NCHW", intermediates_only=True
        )
        return {"out": self.decoder(torch.cat(feats, dim=1), (H, W))}

    def backbone_parameters(self):
        return self.encoder.parameters()


def make_dino(variant: str):
    """Return a constructor bound to one variant (used by the model registry)."""

    def _ctor(num_classes: int = NUM_CLASSES, pretrained: bool = True, **_):   # extra kwargs (input_hw) ignored
        return DinoSeg(variant, num_classes=num_classes, pretrained=pretrained)

    _ctor.__name__ = f"DinoSeg[{variant}]"
    return _ctor
