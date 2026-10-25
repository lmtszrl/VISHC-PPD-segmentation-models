#!/usr/bin/env python3
"""Annotation-consistency statistics of the released polygons.

All quantities are computed from the LabelMe JSON files of the release:
  * region area fractions (area of each region / area of the union of all regions)
  * left-right agreement of area fractions within one examination
  * platform-treadmill agreement of area fractions for the same foot
  * protocol-conformance measures: width ratio of P4 to P5 at the metatarsal level, and the
    angles between the heel line (P8 | midfoot) and the forefoot-midfoot line and the foot axis
  * pairwise overlaps between neighbouring polygons and vertices outside the image bounds

Usage:
    python scripts/annotation_consistency.py --dataset_root "/path/to/VISHC-PPD-dataset"
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

import os

DATASET = Path(os.environ.get("VISHC_PPD_DATASET", "."))     # dataset root; set by the caller or the CLI
ALL = DATASET / "Full data (unclassified)"
REGIONS = [f"P{i}" for i in range(1, 9)]


def set_dataset_root(root):
    global DATASET, ALL
    DATASET = Path(root)
    ALL = DATASET / "Full data (unclassified)"


def _poly(points):
    p = Polygon(points)
    return p if p.is_valid else make_valid(p)


def load(stem):
    j = json.load(open(ALL / f"{stem}.json", encoding="utf-8"))
    polys = {s["label"]: _poly(s["points"]) for s in j["shapes"]}
    return j, polys


def principal_axis(polys):
    """Unit vector of the foot axis (heel -> toes) from the PCA of the footprint outline."""
    union = unary_union(list(polys.values()))
    pts = np.array(union.convex_hull.exterior.coords)
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    ax = vt[0]
    if "P8" in polys:                     # orient from heel towards the toes
        heel = np.array(polys["P8"].centroid.coords[0])
        if np.dot(c - heel, ax) < 0:
            ax = -ax
    return ax / np.linalg.norm(ax), c


def shared_boundary_angle(a, b, axis):
    """Angle (deg) between the boundary shared by polygons a and b and the perpendicular to the axis."""
    inter = a.buffer(1.5).intersection(b.buffer(1.5))
    if inter.is_empty:
        return None
    pts = []
    geoms = inter.geoms if hasattr(inter, "geoms") else [inter]
    for g in geoms:
        if g.geom_type == "Polygon":
            pts.extend(g.exterior.coords)
    pts = np.array(pts)
    if len(pts) < 4:
        return None
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    d = vt[0]
    perp = np.array([-axis[1], axis[0]])
    cosang = abs(np.dot(d, perp)) / np.linalg.norm(d)
    return math.degrees(math.acos(min(1.0, cosang)))


def width_across_axis(poly, axis, centre):
    """Extent of a polygon measured perpendicular to the axis."""
    perp = np.array([-axis[1], axis[0]])
    pts = np.array(poly.exterior.coords) if poly.geom_type == "Polygon" else np.array(unary_union(poly).convex_hull.exterior.coords)
    proj = (pts - centre) @ perp
    return proj.max() - proj.min()


def per_image(stem):
    j, polys = load(stem)
    out = {"stem": stem, "n": len(polys)}
    union = unary_union(list(polys.values()))
    tot = union.area
    out["total_area"] = tot
    for r in REGIONS:
        out[f"frac_{r}"] = polys[r].area / tot if r in polys else np.nan
    ax, c = principal_axis(polys)
    mid = [polys[r] for r in ("P6", "P7") if r in polys]
    fore = [polys[r] for r in ("P3", "P4", "P5") if r in polys]
    out["heel_angle"] = shared_boundary_angle(polys["P8"], unary_union(mid), ax) if mid else np.nan
    out["fore_angle"] = shared_boundary_angle(unary_union(fore), unary_union(mid), ax) if mid else np.nan
    if "P4" in polys and "P5" in polys:
        out["w45_ratio"] = width_across_axis(polys["P4"], ax, c) / width_across_axis(polys["P5"], ax, c)
    else:
        out["w45_ratio"] = np.nan
    # overlaps between polygon pairs
    ov = []
    labs = list(polys)
    for i in range(len(labs)):
        for k in range(i + 1, len(labs)):
            a = polys[labs[i]].intersection(polys[labs[k]]).area
            if a > 0:
                ov.append((labs[i], labs[k], a, a / min(polys[labs[i]].area, polys[labs[k]].area)))
    out["overlaps"] = ov
    # out-of-bounds vertices
    W, H = j["imageWidth"], j["imageHeight"]
    oob = 0.0
    for s in j["shapes"]:
        p = np.array(s["points"])
        d = np.maximum.reduce([np.maximum(-p[:, 0], 0), np.maximum(p[:, 0] - W, 0), np.maximum(-p[:, 1], 0), np.maximum(p[:, 1] - H, 0)])
        oob = max(oob, d.max())
    out["max_oob"] = oob
    return out


def compute():
    meta = {r["image_id"]: r for r in csv.DictReader(open(DATASET / "VISHC-PPD_metadata.csv", encoding="utf-8"))}
    rows = {s: per_image(s) for s in sorted(meta)}
    return meta, rows


def paired_differences(meta, rows):
    """Left-right pairs within an examination and platform-treadmill pairs of the same foot."""
    by_exam = defaultdict(dict)
    by_foot = defaultdict(dict)
    for s, m in meta.items():
        by_exam[(m["subject_id"], m["device"], m["trial"])][m["foot_side"]] = s
        by_foot[(m["subject_id"], m["foot_side"])].setdefault(m["device"], s)   # first trial per device
    lr, dev = [], []
    for k, d in by_exam.items():
        if "L" in d and "R" in d:
            lr.append((d["L"], d["R"]))
    for k, d in by_foot.items():
        if "freegait" in d and "treadmill" in d:
            dev.append((d["freegait"], d["treadmill"]))

    def absdiff(pairs):
        out = {}
        for r in REGIONS:
            v = [abs(rows[a][f"frac_{r}"] - rows[b][f"frac_{r}"]) for a, b in pairs
                 if not (np.isnan(rows[a][f"frac_{r}"]) or np.isnan(rows[b][f"frac_{r}"]))]
            out[r] = np.array(v)
        return out
    return lr, dev, absdiff(lr), absdiff(dev)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Annotation-consistency statistics of a VISHC-PPD dataset copy.")
    ap.add_argument("--dataset_root", required=True)
    set_dataset_root(ap.parse_args().dataset_root)
    meta, rows = compute()
    lr, dev, dlr, ddev = paired_differences(meta, rows)
    print("images", len(rows), "L-R pairs", len(lr), "platform-treadmill foot pairs", len(dev))
    for r in REGIONS:
        f = np.array([rows[s][f"frac_{r}"] for s in rows]); f = f[~np.isnan(f)]
        print(f"{r}: frac {f.mean():.3f} ± {f.std():.3f} | L-R |d| {np.median(dlr[r]):.3f} (90% {np.percentile(dlr[r], 90):.3f}) | dev |d| {np.median(ddev[r]):.3f} (90% {np.percentile(ddev[r], 90):.3f})")
    for key in ("heel_angle", "fore_angle", "w45_ratio"):
        v = np.array([rows[s][key] for s in rows], float); v = v[~np.isnan(v)]
        print(key, f"n={len(v)} median {np.median(v):.2f} IQR {np.percentile(v, 25):.2f}-{np.percentile(v, 75):.2f} 5-95% {np.percentile(v, 5):.2f}-{np.percentile(v, 95):.2f}")
    ov = [o for s in rows for o in rows[s]["overlaps"]]
    areas = np.array([o[2] for o in ov]); fr = np.array([o[3] for o in ov])
    per_img = np.array([sum(o[2] for o in rows[s]["overlaps"]) / rows[s]["total_area"] for s in rows])
    print(f"overlapping pairs {len(ov)}; area px2 median {np.median(areas):.1f} 95% {np.percentile(areas, 95):.1f} max {areas.max():.1f}; "
          f"fraction of smaller region median {np.median(fr)*100:.2f}% 95% {np.percentile(fr, 95)*100:.2f}% max {fr.max()*100:.1f}%; "
          f"total overlap per image / foot area: mean {per_img.mean()*100:.2f}% max {per_img.max()*100:.2f}%")
    oob = [rows[s]["max_oob"] for s in rows]
    print("images with out-of-bounds vertices", sum(1 for v in oob if v > 0), "max", round(max(oob), 2), "px")


if __name__ == "__main__":
    main()
