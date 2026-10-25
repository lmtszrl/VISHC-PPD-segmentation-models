"""
vishc_ppd — reference code for the VISHC-PPD plantar-pressure segmentation benchmark.

Sub-packages
------------
data    : LabelMe annotation I/O, deterministic rasterisation, dataset / split handling
models  : five segmentation architectures behind one interface (``model(x)["out"]``)
metrics : confusion-matrix based segmentation metrics
losses  : cross-entropy + soft-Dice compound loss
engine  : training / evaluation loops shared by the CLI scripts
utils   : seeding
"""

__version__ = "1.0.0"

from .constants import (  # noqa: F401
    NUM_CLASSES,
    REGION_IDS,
    REGION_LABELS,
    REGION_LONG_NAMES,
    REGION_PALETTE,
)

__all__ = ["NUM_CLASSES", "REGION_IDS", "REGION_LABELS", "REGION_LONG_NAMES", "REGION_PALETTE", "__version__"]
