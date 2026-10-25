#!/usr/bin/env python3
"""
Generate the reference training/test split of VISHC-PPD from the metadata table.

The split is drawn at the participant level: all images of one participant
(both feet, both devices, all trials) fall on the same side.  Among random
participant-level draws that yield exactly ``--n-test`` test images, the draw
whose test subset best matches the overall distribution of device, sex, age
group, foot type and foot side (smallest sum of absolute differences of the
proportions) is kept.

Usage:
    python scripts/make_reference_split.py VISHC-PPD_metadata.csv --out train_test_split.csv
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

STRATA = ("device", "sex", "age_group", "foot_type", "foot_side")


def proportions(rows, key):
    c = Counter(r[key] for r in rows)
    n = sum(c.values())
    return {k: v / n for k, v in c.items()}


def distance(test_rows, all_rows):
    d = 0.0
    for key in STRATA:
        p_all = proportions(all_rows, key)
        p_test = proportions(test_rows, key)
        d += sum(abs(p_all[k] - p_test.get(k, 0.0)) for k in p_all)
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("metadata", type=Path, help="VISHC-PPD_metadata.csv (the split column is ignored)")
    ap.add_argument("--out", type=Path, default=Path("train_test_split.csv"))
    ap.add_argument("--n-test", type=int, default=108, help="number of test images (default 108, about 20 %% of the release)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--draws", type=int, default=200_000, help="number of random draws to examine")
    args = ap.parse_args()

    with args.metadata.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_subject = defaultdict(list)
    for r in rows:
        by_subject[r["subject_id"]].append(r)
    subjects = sorted(by_subject)
    rng = random.Random(args.seed)

    best = None
    for _ in range(args.draws):
        order = subjects[:]
        rng.shuffle(order)
        test, n = [], 0
        for sid in order:
            k = len(by_subject[sid])
            if n + k > args.n_test:
                continue
            test.append(sid)
            n += k
            if n == args.n_test:
                break
        if n != args.n_test:
            continue
        test_rows = [r for sid in test for r in by_subject[sid]]
        d = distance(test_rows, rows)
        if best is None or d < best[0]:
            best = (d, sorted(test))
    if best is None:
        raise SystemExit("no draw reached the requested number of test images")

    test_subjects = set(best[1])
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "subject_id", "split"])
        for r in rows:
            w.writerow([r["image_id"], r["subject_id"], "test" if r["subject_id"] in test_subjects else "train"])
    n_test = sum(1 for r in rows if r["subject_id"] in test_subjects)
    print(f"wrote {args.out}: {len(rows) - n_test} train / {n_test} test images, "
          f"{len(subjects) - len(test_subjects)} / {len(test_subjects)} participants, distance {best[0]:.4f}")
    print("test participants:", ", ".join(best[1]))


if __name__ == "__main__":
    main()
