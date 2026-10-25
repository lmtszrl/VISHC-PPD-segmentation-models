"""Project-wide constants: class indices, names, colours."""

from __future__ import annotations

import numpy as np

# Class index 0 is background; 1..8 are the eight plantar regions P1..P8.
NUM_CLASSES: int = 9
REGION_IDS: tuple[int, ...] = tuple(range(1, 9))
REGION_LABELS: tuple[str, ...] = tuple(f"P{k}" for k in REGION_IDS)
CLASS_NAMES: tuple[str, ...] = ("background",) + REGION_LABELS

REGION_LONG_NAMES: dict[str, str] = {
    "P1": "Hallux",
    "P2": "Lesser toes",
    "P3": "1st metatarsal",
    "P4": "Central metatarsals (2nd-4th)",
    "P5": "Lateral metatarsal (5th)",
    "P6": "Medial midfoot",
    "P7": "Lateral midfoot",
    "P8": "Heel",
}

# Display palette (index 0 = background). Chosen for legibility in print.
REGION_PALETTE: np.ndarray = np.array(
    [
        [30, 30, 30],     # 0 background
        [220, 50, 50],    # P1 hallux            - red
        [255, 140, 0],    # P2 lesser toes       - orange
        [200, 200, 0],    # P3 1st metatarsal    - olive-yellow
        [50, 180, 50],    # P4 central metas     - green
        [0, 200, 200],    # P5 lateral meta      - cyan
        [70, 130, 220],   # P6 medial midfoot    - cornflower blue
        [30, 60, 200],    # P7 lateral midfoot   - blue
        [150, 50, 200],   # P8 heel              - purple
    ],
    dtype=np.uint8,
)

# ImageNet statistics used by every backbone in the benchmark.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# Default network input size (H, W). Plantar prints are roughly 2:1 tall.
DEFAULT_INPUT_HW: tuple[int, int] = (512, 256)

# File-name pattern of the released dataset: S0001_freegait_L[_trial2]
IMAGE_ID_PATTERN = r"^(S\d{4})_(freegait|treadmill)_([LR])(?:_trial(\d+))?$"
