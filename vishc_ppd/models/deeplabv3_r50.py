"""
DeepLabV3 with a ResNet-50 backbone and an FCN auxiliary head (torchvision).

Chen et al., "Rethinking Atrous Convolution for Semantic Image Segmentation",
arXiv 2017.  The torchvision implementation is initialised from COCO/VOC
weights (ImageNet backbone) and both classification heads are replaced.
"""

from __future__ import annotations

from typing import Dict

import torch
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead

from ..constants import NUM_CLASSES
from .base import SegmentationModel


class DeepLabV3R50(SegmentationModel):
    display_name = "DeepLabV3 (R50)"

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True, **_):
        super().__init__()
        weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
        # weights_backbone=None: the backbone comes with the segmentation weights (or stays random)
        self.model = deeplabv3_resnet50(weights=weights, weights_backbone=None, aux_loss=True)
        self.model.classifier = DeepLabHead(2048, num_classes)
        self.model.aux_classifier = FCNHead(1024, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.model(x)
        if not self.training:
            out.pop("aux", None)
        return out

    def backbone_parameters(self):
        return self.model.backbone.parameters()
