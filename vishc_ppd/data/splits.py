"""
Sample discovery and split handling.

Two ways of pointing the code at data are supported:

* **Dataset root** (recommended) — the released ``VISHC-PPD-dataset``
  folder.  Images and JSONs are read from ``Full data (unclassified)/`` and the
  train/test membership from ``train_test_split.csv``; subject IDs come from
  the file names, so validation splits can be made subject-disjoint.

* **Plain folder** — any folder with ``<stem>.jpg`` (+ optional ``<stem>.json``)
  pairs, e.g. ``train data/`` / ``test data/`` of the release, or a folder of
  new images for inference.
"""

from __future__ import annotations

import csv
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..constants import IMAGE_ID_PATTERN

__all__ = [
    "Sample",
    "IMAGES_DIRNAME",
    "SPLIT_FILENAME",
    "METADATA_FILENAME",
    "subject_id_from_stem",
    "collect_from_folder",
    "collect_from_dataset",
    "read_split_table",
    "read_metadata",
    "subject_level_split",
]

IMAGES_DIRNAME = "Full data (unclassified)"
SPLIT_FILENAME = "train_test_split.csv"
METADATA_FILENAME = "VISHC-PPD_metadata.csv"
_IMG_EXTS = {".jpg", ".jpeg", ".png"}
_ID_RE = re.compile(IMAGE_ID_PATTERN)


@dataclass(frozen=True)
class Sample:
    image_path: Path
    json_path: Optional[Path]      # None when no annotation is available
    image_id: str                  # file stem, e.g. S0005_freegait_L
    subject_id: str                # e.g. S0005 (falls back to the stem)

    @property
    def has_annotation(self) -> bool:
        return self.json_path is not None and self.json_path.exists()


def subject_id_from_stem(stem: str) -> str:
    """``S0005_freegait_L_trial2`` → ``S0005``; unknown patterns return the stem."""
    m = _ID_RE.match(stem)
    return m.group(1) if m else stem


def _make_sample(img: Path) -> Sample:
    jp = img.with_suffix(".json")
    return Sample(
        image_path=img,
        json_path=jp if jp.exists() else None,
        image_id=img.stem,
        subject_id=subject_id_from_stem(img.stem),
    )


# ── Plain folder ─────────────────────────────────────────────────────────────

def collect_from_folder(folder: Path | str, require_json: bool = False) -> List[Sample]:
    """All image files in ``folder`` (non-recursive), sorted by name."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"not a directory: {folder}")
    imgs = sorted(p for p in folder.iterdir() if p.suffix.lower() in _IMG_EXTS and not p.name.startswith("."))
    samples = [_make_sample(p) for p in imgs]
    if require_json:
        samples = [s for s in samples if s.has_annotation]
    if not samples:
        raise RuntimeError(f"no {'annotated ' if require_json else ''}images found in {folder}")
    return samples


# ── Dataset root ─────────────────────────────────────────────────────────────

def read_split_table(root: Path | str) -> Dict[str, str]:
    """``{image_id: 'train' | 'test'}`` from ``train_test_split.csv``."""
    path = Path(root) / SPLIT_FILENAME
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or "image_id" not in rows[0] or "split" not in rows[0]:
        raise RuntimeError(f"{path} must have columns image_id, split")
    return {r["image_id"]: r["split"] for r in rows}


def read_metadata(root: Path | str) -> Dict[str, Dict[str, str]]:
    """``{image_id: row}`` from ``VISHC-PPD_metadata.csv``."""
    path = Path(root) / METADATA_FILENAME
    with path.open(newline="", encoding="utf-8") as f:
        return {r["image_id"]: r for r in csv.DictReader(f)}


def collect_from_dataset(
    root: Path | str,
    split: str,
    images_dirname: str = IMAGES_DIRNAME,
) -> List[Sample]:
    """
    Samples of one split (``'train'`` / ``'test'`` / ``'all'``) from a dataset
    root, resolved through ``train_test_split.csv``.
    """
    root = Path(root)
    table = read_split_table(root)
    if split != "all" and split not in set(table.values()):
        raise ValueError(f"split {split!r} not present in {SPLIT_FILENAME}")
    img_dir = root / images_dirname
    samples: List[Sample] = []
    missing: List[str] = []
    for image_id, sp in sorted(table.items()):
        if split != "all" and sp != split:
            continue
        img = img_dir / f"{image_id}.jpg"
        if not img.exists():
            missing.append(image_id)
            continue
        samples.append(_make_sample(img))
    if missing:
        raise FileNotFoundError(f"{len(missing)} images listed in {SPLIT_FILENAME} are missing "
                                f"from {img_dir} (first: {missing[:3]})")
    return samples


# ── Subject-level validation split ───────────────────────────────────────────

def subject_level_split(
    samples: Sequence[Sample],
    val_fraction: float,
    seed: int,
) -> Tuple[List[Sample], List[Sample]]:
    """
    Hold out roughly ``val_fraction`` of the *images* for validation such that
    no subject appears in both parts.  Deterministic for a given seed.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    by_subject: Dict[str, List[Sample]] = defaultdict(list)
    for s in samples:
        by_subject[s.subject_id].append(s)
    subjects = sorted(by_subject)
    rng = random.Random(seed)
    rng.shuffle(subjects)

    target = val_fraction * len(samples)
    val_subjects: List[str] = []
    n_val = 0
    for sid in subjects:
        if n_val >= target:
            break
        val_subjects.append(sid)
        n_val += len(by_subject[sid])
    val_set = set(val_subjects)
    train = [s for s in samples if s.subject_id not in val_set]
    val = [s for s in samples if s.subject_id in val_set]
    if not train or not val:
        raise RuntimeError("subject-level split produced an empty partition")
    return train, val


def summarize(samples: Iterable[Sample]) -> str:
    samples = list(samples)
    n_subj = len({s.subject_id for s in samples})
    n_ann = sum(1 for s in samples if s.has_annotation)
    return f"{len(samples)} images / {n_subj} subjects ({n_ann} annotated)"
