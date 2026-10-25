"""
Model registry.

>>> from vishc_ppd.models import get_model, MODEL_NAMES
>>> model = get_model("unet", num_classes=9, pretrained=True)
>>> logits = model(x)["out"]          # (N, 9, H, W)
"""

from __future__ import annotations

from typing import Callable, Dict

from .base import SegmentationModel
from .deeplabv3_r50 import DeepLabV3R50
from .dino import DINO_VARIANTS, DinoSeg, make_dino
from .segformer_b2 import SegFormerB2
from .swin_unet import SwinUNet
from .unet import UNet

_REGISTRY: Dict[str, Callable[..., SegmentationModel]] = {
    "deeplabv3_r50": DeepLabV3R50,
    "unet": UNet,
    "segformer_b2": SegFormerB2,
    "swin_unet": SwinUNet,
    **{name: make_dino(name) for name in DINO_VARIANTS},
}

MODEL_NAMES = list(_REGISTRY)

# Short names for tables / figures (no line breaks; scripts wrap as needed)
DISPLAY_NAMES: Dict[str, str] = {
    "deeplabv3_r50": "DeepLabV3 (R50)",
    "unet": "U-Net (R34)",
    "segformer_b2": "SegFormer-B2",
    "swin_unet": "Swin-UNet (T)",
    **{name: v.display for name, v in DINO_VARIANTS.items()},
}


def get_model(name: str, num_classes: int = 9, pretrained: bool = True, **kwargs) -> SegmentationModel:
    """
    Instantiate a benchmark model by registry name.

    ``input_hw=(H, W)`` may be passed for models whose architecture depends on
    the input size (Swin-UNet); other models ignore it.
    """
    if name not in _REGISTRY:
        raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
    return _REGISTRY[name](num_classes=num_classes, pretrained=pretrained, **kwargs)


__all__ = ["get_model", "MODEL_NAMES", "DISPLAY_NAMES", "SegmentationModel", "DinoSeg", "DINO_VARIANTS",
           "DeepLabV3R50", "UNet", "SegFormerB2", "SwinUNet"]
