"""
LabelMe annotation I/O and deterministic polygon rasterisation.

The released VISHC-PPD annotations are LabelMe 5.x JSON files in which every
``shapes`` entry is a polygon labelled exactly ``P1`` … ``P8`` (each label at
most once per image; a region that is genuinely absent from the footprint has
no polygon).  Two properties of the annotations require an explicit rule when
converting polygons to a label mask:

1. **Region overlap.**  Neighbouring regions (mostly P6/P7 in the midfoot and
   P4/P5 in the forefoot) may overlap by a few percent.  ``polygons_to_mask``
   paints regions in *descending order of polygon area*, so that a smaller
   region always wins over a larger one where they overlap.  This is
   independent of the order of shapes in the JSON file and therefore fully
   reproducible.

2. **Self-intersecting outlines.**  A small number of polygons (50 in the
   release) have a minor "bow-tie" where two edges cross within a few pixels.
   Such polygons are repaired with ``shapely.make_valid`` before filling, and
   the repaired geometry (polygon or multi-polygon) is rasterised with the
   non-zero rule.  Without this step, different rasterisers (OpenCV, PIL,
   scikit-image, …) disagree on the affected pixels.

Both rules are documented in the dataset README and in the manuscript.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.validation import make_valid

from ..constants import NUM_CLASSES, REGION_LABELS

__all__ = [
    "parse_region_id",
    "load_labelme",
    "polygons_to_mask",
    "load_mask",
    "mask_to_polygons",
    "write_labelme_json",
    "AnnotationError",
]

_LABEL_RE = re.compile(r"^P([1-8])$")


class AnnotationError(ValueError):
    """Raised when a JSON file violates the VISHC-PPD annotation contract."""


# ── Label parsing ────────────────────────────────────────────────────────────

def parse_region_id(label: Optional[str], strict: bool = True) -> Optional[int]:
    """
    Map a LabelMe label to a class index 1..8.

    With ``strict=True`` (the default, matching the released dataset) only the
    exact strings ``"P1"`` … ``"P8"`` are accepted; anything else raises
    :class:`AnnotationError`.  With ``strict=False`` the parser tolerates
    non-canonical variants such as ``"p3"`` or ``"P4e"`` and returns ``None`` for
    unrelated labels (useful when importing raw LabelMe exports).
    """
    if label is None:
        if strict:
            raise AnnotationError("shape without a label")
        return None
    text = label.strip()
    m = _LABEL_RE.match(text)
    if m:
        return int(m.group(1))
    if strict:
        raise AnnotationError(f"unexpected label {label!r}; expected one of {REGION_LABELS}")
    m = re.search(r"[Pp]\s*([1-8])", text)
    return int(m.group(1)) if m else None


# ── JSON loading ─────────────────────────────────────────────────────────────

def load_labelme(json_path: Path | str) -> Dict[str, Any]:
    """Read a LabelMe JSON file (UTF-8) and return the parsed dictionary."""
    return json.loads(Path(json_path).read_text(encoding="utf-8"))


def _polygon_shapes(ann: Dict[str, Any], strict: bool) -> List[tuple[int, np.ndarray]]:
    """Return ``[(class_id, points[N,2] float32), …]`` for every valid polygon."""
    out: List[tuple[int, np.ndarray]] = []
    seen: set[int] = set()
    for sh in ann.get("shapes", []):
        if sh.get("shape_type", "polygon") != "polygon":
            if strict:
                raise AnnotationError(f"non-polygon shape of type {sh.get('shape_type')!r}")
            continue
        cid = parse_region_id(sh.get("label"), strict=strict)
        if cid is None:
            continue
        pts = np.asarray(sh.get("points", []), dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] < 3:
            if strict:
                raise AnnotationError(f"polygon {sh.get('label')!r} has fewer than 3 points")
            continue
        if strict and cid in seen:
            raise AnnotationError(f"label P{cid} appears more than once")
        seen.add(cid)
        out.append((cid, pts))
    return out


# ── Rasterisation ────────────────────────────────────────────────────────────

def _valid_geometries(pts: np.ndarray) -> List[np.ndarray]:
    """
    Return a list of simple exterior rings (each ``[N,2]``) that together
    represent the polygon.  Self-intersecting inputs are repaired with
    ``shapely.make_valid``; valid inputs are returned unchanged.
    """
    poly = Polygon(pts)
    if poly.is_valid:
        return [pts]
    fixed = make_valid(poly)
    rings: List[np.ndarray] = []
    geoms = fixed.geoms if hasattr(fixed, "geoms") else [fixed]
    for g in geoms:
        if isinstance(g, Polygon) and g.area > 0:
            rings.append(np.asarray(g.exterior.coords, dtype=np.float32)[:-1])
        elif isinstance(g, MultiPolygon):
            for gg in g.geoms:
                if gg.area > 0:
                    rings.append(np.asarray(gg.exterior.coords, dtype=np.float32)[:-1])
        # lines / points produced by make_valid carry no area and are dropped
    return rings or [pts]


def _fill(mask: np.ndarray, rings: Sequence[np.ndarray], value: int) -> None:
    h, w = mask.shape
    for ring in rings:
        pts = np.round(ring).astype(np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        cv2.fillPoly(mask, [pts], int(value))


def polygons_to_mask(
    ann: Dict[str, Any],
    h: Optional[int] = None,
    w: Optional[int] = None,
    strict: bool = True,
) -> np.ndarray:
    """
    Rasterise a LabelMe annotation to an ``(H, W)`` uint8 label mask.

    Pixel values: 0 = background, 1..8 = P1..P8.  Regions are painted in
    descending order of polygon area (smaller regions win on overlap); each
    polygon is validated / repaired first (see module docstring).

    ``h`` / ``w`` default to ``imageHeight`` / ``imageWidth`` from the JSON.
    """
    h = int(h if h is not None else ann["imageHeight"])
    w = int(w if w is not None else ann["imageWidth"])
    mask = np.zeros((h, w), dtype=np.uint8)

    shapes = _polygon_shapes(ann, strict=strict)
    # Larger polygons first, smaller on top.  Ties broken by class id so the
    # result never depends on JSON order.
    shapes.sort(key=lambda t: (-abs(Polygon(t[1]).area), t[0]))
    for cid, pts in shapes:
        _fill(mask, _valid_geometries(pts), cid)
    return mask


def load_mask(json_path: Path | str, h: Optional[int] = None, w: Optional[int] = None,
              strict: bool = True) -> np.ndarray:
    """Convenience wrapper: read a JSON file and return its rasterised mask."""
    return polygons_to_mask(load_labelme(json_path), h, w, strict=strict)


# ── Mask → polygons (for exporting predictions) ─────────────────────────────

def mask_to_polygons(
    mask: np.ndarray,
    class_id: int,
    min_area: float = 30.0,
    epsilon_ratio: float = 0.003,
    keep_top_k: int = 1,
) -> List[List[List[float]]]:
    """
    Extract polygon contours of ``class_id`` from a label mask.

    Returns up to ``keep_top_k`` polygons ordered by area, each as
    ``[[x, y], …]`` floats; contours are simplified with Douglas-Peucker.
    """
    binary = (mask == class_id).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    polys: List[List[List[float]]] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        eps = max(1.0, epsilon_ratio * cv2.arcLength(cnt, True))
        approx = cv2.approxPolyDP(cnt, eps, True)
        pts = approx[:, 0, :].astype(np.float32).tolist()
        if len(pts) >= 3:
            polys.append(pts)
        if len(polys) >= keep_top_k:
            break
    return polys


def write_labelme_json(
    out_path: Path | str,
    image_name: str,
    pred_mask: np.ndarray,
    version: str = "5.0.2",
) -> None:
    """
    Write a LabelMe-compatible JSON (``imageData`` = null) containing the
    P1..P8 polygons extracted from ``pred_mask``.
    """
    h, w = pred_mask.shape[:2]
    shapes = []
    for k in range(1, NUM_CLASSES):
        for pts in mask_to_polygons(pred_mask, k):
            shapes.append({
                "label": f"P{k}",
                "points": pts,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {},
            })
    payload = {
        "version": version,
        "flags": {},
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": int(h),
        "imageWidth": int(w),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
