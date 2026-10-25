#!/usr/bin/env python3
"""
Validate a copy of the VISHC-PPD release against its documented contract.

Checks
------
* every image_id in train_test_split.csv and VISHC-PPD_metadata.csv exists as
  <image_id>.jpg + .json in "Full data (unclassified)" and vice-versa;
* file names follow S####_<device>_<L|R>[_trialN]; metadata columns agree
  with the file name (subject, device, side, trial); split is subject-level;
* every JSON is LabelMe 5.x with imagePath = own file name, imageData = null,
  imageWidth/Height equal to the JPG, polygons only, labels exactly P1..P8
  (each at most once), ≥ 3 points; regions absent from the footprint are
  reported (5 documented cases) — the check fails only for *unexpected*
  absences unless --allow_missing_regions is given;
* every JPG decodes; self-intersecting polygons are counted (informational).

Exit status is 0 when all checks pass, 1 otherwise.

Usage:  python validate_dataset.py --dataset_root "/path/to/VISHC-PPD-dataset"
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vishc_ppd.constants import IMAGE_ID_PATTERN, REGION_LABELS  # noqa: E402
from vishc_ppd.data import IMAGES_DIRNAME, METADATA_FILENAME, SPLIT_FILENAME, AnnotationError, load_labelme  # noqa: E402
from vishc_ppd.data.annotations import _polygon_shapes  # noqa: E402

# Images in which a region is genuinely absent from the footprint (documented in the release README).
DOCUMENTED_ABSENT = {
    "S0001_freegait_L": {"P2"},
    "S0147_freegait_L": {"P6"},
    "S0148_freegait_L": {"P6"},
    "S0148_freegait_R": {"P6"},
    "S0148_treadmill_L": {"P6"},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--allow_missing_regions", action="store_true",
                    help="Do not fail on regions missing from images outside the documented list.")
    args = ap.parse_args()
    root = Path(args.dataset_root)
    img_dir = root / IMAGES_DIRNAME
    problems: list[str] = []
    info: list[str] = []

    def fail(msg: str):
        problems.append(msg)

    # ── tables ────────────────────────────────────────────────────────────
    with (root / SPLIT_FILENAME).open(newline="", encoding="utf-8") as f:
        split = {r["image_id"]: r for r in csv.DictReader(f)}
    with (root / METADATA_FILENAME).open(newline="", encoding="utf-8") as f:
        meta = {r["image_id"]: r for r in csv.DictReader(f)}
    files = {p.stem for p in img_dir.glob("*.jpg")}
    jsons = {p.stem for p in img_dir.glob("*.json")}
    for name, ids in (("split table", set(split)), ("metadata", set(meta)), ("json files", jsons)):
        if ids != files:
            fail(f"{name} ({len(ids)}) and jpg files ({len(files)}) differ: {sorted(ids ^ files)[:5]}")
    pat = re.compile(IMAGE_ID_PATTERN)
    subj_split = defaultdict(set)
    for iid, m in meta.items():
        mm = pat.match(iid)
        if not mm:
            fail(f"{iid}: file name does not match the naming pattern")
            continue
        sid, dev, side, trial = mm.groups()
        if (m["subject_id"], m["device"], m["foot_side"], m["trial"] or None) != (sid, dev, side, trial):
            fail(f"{iid}: metadata columns disagree with the file name")
        if iid in split and split[iid]["split"] != m["split"]:
            fail(f"{iid}: split differs between the two tables")
        subj_split[sid].add(m["split"])
        if m["sex"] not in ("male", "female") or m["foot_type"] not in ("Normal", "Flat", "Hollow"):
            fail(f"{iid}: invalid sex / foot_type value")
        if m["age"] == "" or m["age_group"] != ("le12" if int(m["age"]) <= 12 else "gt12"):
            fail(f"{iid}: age and age_group inconsistent")
    leaks = [s for s, v in subj_split.items() if len(v) > 1]
    if leaks:
        fail(f"subjects present in both splits: {leaks}")
    info.append(f"{len(files)} images, {len(subj_split)} subjects, split counts {Counter(m['split'] for m in meta.values())}")

    # ── annotations ──────────────────────────────────────────────────────
    absent: dict[str, set] = {}
    self_intersecting = 0
    for stem in sorted(files):
        jp, ip = img_dir / f"{stem}.json", img_dir / f"{stem}.jpg"
        try:
            im = Image.open(ip)
            im.load()
        except Exception as e:  # noqa: BLE001
            fail(f"{stem}.jpg cannot be decoded ({e})")
            continue
        if not jp.exists():
            continue
        ann = load_labelme(jp)
        if set(ann) != {"version", "flags", "shapes", "imagePath", "imageData", "imageHeight", "imageWidth"}:
            fail(f"{stem}.json: unexpected top-level keys {sorted(ann)}")
        if ann.get("imagePath") != f"{stem}.jpg":
            fail(f"{stem}.json: imagePath is {ann.get('imagePath')!r}")
        if ann.get("imageData") is not None:
            fail(f"{stem}.json: imageData is not null")
        if (ann.get("imageWidth"), ann.get("imageHeight")) != im.size:
            fail(f"{stem}.json: imageWidth/Height {ann.get('imageWidth')}x{ann.get('imageHeight')} != jpg {im.size}")
        try:
            shapes = _polygon_shapes(ann, strict=True)
        except AnnotationError as e:
            fail(f"{stem}.json: {e}")
            continue
        present = {f"P{c}" for c, _ in shapes}
        missing = set(REGION_LABELS) - present
        if missing:
            absent[stem] = missing
            if missing != DOCUMENTED_ABSENT.get(stem, set()) and not args.allow_missing_regions:
                fail(f"{stem}.json: region(s) {sorted(missing)} missing (not in the documented list)")
        for _, pts in shapes:
            if not Polygon(pts).is_valid:
                self_intersecting += 1
    documented_ok = all(absent.get(k) == v for k, v in DOCUMENTED_ABSENT.items())
    if not documented_ok:
        fail("documented absent-region list does not match the data")
    info.append(f"{len(absent)} images with an absent region: {absent}")
    info.append(f"{self_intersecting} self-intersecting polygons (repaired automatically at rasterisation time)")

    # ── report ───────────────────────────────────────────────────────────
    print("\n".join(f"  {i}" for i in info))
    if problems:
        print(f"\nFAILED — {len(problems)} problem(s):")
        for p in problems[:50]:
            print("  -", p)
        return 1
    print("\nOK — dataset passes all checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
