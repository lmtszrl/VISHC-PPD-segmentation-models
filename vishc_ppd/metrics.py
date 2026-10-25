"""
Confusion-matrix based segmentation metrics.

The confusion matrix is accumulated over **all** pixels, background included
(rows = ground truth, columns = prediction).  Metrics are then *reported* for
the eight foreground regions only, so that:

* a foreground pixel predicted as background counts as a false negative, and
* a background pixel predicted as a region counts as a false positive of that
  region (it enlarges the union and lowers IoU / precision).

Dropping background pixels before accumulating would ignore the second kind
of error and inflate every score.

Reported metrics (foreground classes P1..P8)
---------------------------------------------
global_acc   fraction of foreground GT pixels that received the correct label
mean_acc     per-class recall averaged over the eight regions
miou         mean IoU over the eight regions
wiou         IoU weighted by the GT pixel frequency of each region
f1           macro-averaged Dice / F1
precision    macro-averaged precision
recall       same as mean_acc (kept for tables)
iou_Pk, acc_Pk, f1_Pk, precision_Pk   per-region values
pixel_acc    plain pixel accuracy over *all* classes incl. background
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from .constants import CLASS_NAMES, NUM_CLASSES

__all__ = [
    "make_empty_confusion",
    "update_confusion",
    "metrics_from_confusion",
    "compute_metrics",
    "per_image_iou",
]


def make_empty_confusion(num_classes: int = NUM_CLASSES, device: str | torch.device = "cpu") -> torch.Tensor:
    return torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)


@torch.no_grad()
def update_confusion(
    conf: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = NUM_CLASSES,
) -> torch.Tensor:
    """Accumulate ``conf[gt, pred]`` counts from integer label tensors of any shape."""
    pred = pred.reshape(-1).to(conf.device)
    target = target.reshape(-1).to(conf.device)
    valid = (target >= 0) & (target < num_classes) & (pred >= 0) & (pred < num_classes)
    idx = target[valid] * num_classes + pred[valid]
    conf += torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return conf


@torch.no_grad()
def metrics_from_confusion(conf: torch.Tensor, background_index: int = 0) -> Dict[str, float]:
    """Derive all scalar and per-region metrics from a full confusion matrix."""
    conf = conf.detach().cpu().double()
    C = conf.shape[0]
    eps = 1e-12

    tp = torch.diag(conf)
    gt = conf.sum(dim=1)     # per-class GT pixels
    pr = conf.sum(dim=0)     # per-class predicted pixels
    fg = torch.ones(C, dtype=torch.bool)
    fg[background_index] = False

    tp_f, gt_f, pr_f = tp[fg], gt[fg], pr[fg]
    union = gt_f + pr_f - tp_f

    recall_c = tp_f / gt_f.clamp(min=eps)
    prec_c = tp_f / pr_f.clamp(min=eps)
    iou_c = tp_f / union.clamp(min=eps)
    f1_c = 2 * tp_f / (gt_f + pr_f).clamp(min=eps)
    freq = gt_f / gt_f.sum().clamp(min=eps)

    # Regions that never occur in the GT of the evaluated set (and were never
    # predicted) are excluded from macro averages instead of counting as 0.
    present = (gt_f + pr_f) > 0

    def macro(v: torch.Tensor) -> float:
        return float(v[present].mean()) if present.any() else float("nan")

    out: Dict[str, float] = {
        "global_acc": float(tp_f.sum() / gt_f.sum().clamp(min=eps)),
        "mean_acc": macro(recall_c),
        "miou": macro(iou_c),
        "wiou": float((freq * iou_c).sum()),
        "f1": macro(f1_c),
        "precision": macro(prec_c),
        "recall": macro(recall_c),
        "pixel_acc": float(tp.sum() / conf.sum().clamp(min=eps)),
    }
    names = [CLASS_NAMES[i] for i in range(C) if fg[i]]
    for j, name in enumerate(names):
        out[f"iou_{name}"] = float(iou_c[j])
        out[f"acc_{name}"] = float(recall_c[j])
        out[f"f1_{name}"] = float(f1_c[j])
        out[f"precision_{name}"] = float(prec_c[j])
    return out


def compute_metrics(pred: np.ndarray, target: np.ndarray, num_classes: int = NUM_CLASSES) -> Dict[str, float]:
    """One-shot metrics from numpy label arrays of identical shape."""
    conf = make_empty_confusion(num_classes)
    update_confusion(conf, torch.from_numpy(pred.astype(np.int64)), torch.from_numpy(target.astype(np.int64)), num_classes)
    return metrics_from_confusion(conf)


def per_image_iou(pred: np.ndarray, target: np.ndarray, num_classes: int = NUM_CLASSES) -> Dict[str, Optional[float]]:
    """
    Per-region IoU for a single image; a region absent from both GT and
    prediction yields ``None`` (undefined) rather than 0 or 1.
    """
    out: Dict[str, Optional[float]] = {}
    for k in range(1, num_classes):
        g = target == k
        p = pred == k
        union = np.logical_or(g, p).sum()
        out[f"iou_{CLASS_NAMES[k]}"] = None if union == 0 else float(np.logical_and(g, p).sum() / union)
    vals = [v for v in out.values() if v is not None]
    out["miou"] = float(np.mean(vals)) if vals else None
    return out
