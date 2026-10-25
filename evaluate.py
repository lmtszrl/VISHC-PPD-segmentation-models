#!/usr/bin/env python3
"""
Evaluate trained checkpoints on the annotated test split and write the
metric tables reported in the manuscript.

Two ways to pass checkpoints:

1. ``--runs_dir runs`` — scans ``runs/<model>/<run>/best.pt`` (one folder per
   seed) and reports mean ± SD over runs of the same model.
2. ``--models m1 m2 … --checkpoints c1 c2 …`` — explicit pairs (single run each).

All metrics are computed at the **native image resolution** from a confusion
matrix that includes the background class (see ``vishc_ppd.metrics``).

Outputs (in --out_dir)
----------------------
metrics_runs.csv          one row per checkpoint (all scalar + per-region metrics)
metrics_summary.csv       one row per model: mean ± SD, #params, inference time
metrics_per_region.csv    per-region IoU / F1 / accuracy per model (mean over runs)
metrics_per_image.csv     per-image mIoU for every checkpoint (for statistics)
metrics_full.json         everything above as JSON (incl. raw confusion matrices)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vishc_ppd.constants import NUM_CLASSES, REGION_LABELS  # noqa: E402
from vishc_ppd.data import Sample, collect_from_dataset, collect_from_folder, load_mask, read_image_rgb  # noqa: E402
from vishc_ppd.engine import Predictor, load_checkpoint, resolve_device  # noqa: E402
from vishc_ppd.metrics import make_empty_confusion, metrics_from_confusion, per_image_iou, update_confusion  # noqa: E402
from vishc_ppd.models import DISPLAY_NAMES, MODEL_NAMES  # noqa: E402

SCALARS = [("global_acc", "Global Acc."), ("mean_acc", "Mean Acc."), ("miou", "mIoU"), ("wiou", "WIoU"),
           ("f1", "F1"), ("precision", "Precision"), ("recall", "Recall")]


# ── Checkpoint discovery ─────────────────────────────────────────────────────

def discover_runs(runs_dir: Path) -> List[Tuple[str, str, Path]]:
    """``[(model, run_name, best.pt), …]`` for runs/<model>/<run>/best.pt (or runs/<model>/best.pt)."""
    found = []
    for model_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        if model_dir.name not in MODEL_NAMES:
            continue
        direct = model_dir / "best.pt"
        if direct.exists():
            found.append((model_dir.name, "run", direct))
        for run_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            ckpt = run_dir / "best.pt"
            if ckpt.exists():
                found.append((model_dir.name, run_dir.name, ckpt))
    if not found:
        raise SystemExit(f"no best.pt found under {runs_dir}")
    return found


# ── Evaluation of one checkpoint ─────────────────────────────────────────────

@torch.no_grad()
def evaluate_checkpoint(model_name: str, ckpt_path: Path, samples: List[Sample], device, size_hw, latency_repeats: int):
    model, ckpt = load_checkpoint(ckpt_path, model_name, device)
    size_hw = resolve_size(size_hw, ckpt)
    predictor = Predictor(model, device, size_hw)
    conf = make_empty_confusion(NUM_CLASSES)
    per_image = []
    times = []
    for s in tqdm(samples, desc=f"  {model_name}", leave=False):
        img = read_image_rgb(s.image_path)
        gt = load_mask(s.json_path, *img.shape[:2])
        pred, ms = predictor.timed(img, repeats=latency_repeats)
        times.append(ms)
        update_confusion(conf, torch.from_numpy(pred.astype(np.int64)), torch.from_numpy(gt.astype(np.int64)))
        per_image.append({"image_id": s.image_id, **per_image_iou(pred, gt)})
    mets = metrics_from_confusion(conf)
    mets["params_M"] = model.count_parameters() / 1e6
    mets["infer_ms"] = float(np.median(times[1:] if len(times) > 1 else times))   # skip warm-up sample
    mets["epoch"] = int(ckpt.get("epoch", -1))
    mets["input_h"], mets["input_w"] = size_hw
    return mets, conf, per_image


def resolve_size(size_hw, ckpt) -> Tuple[int, int]:
    """Use the CLI size when given, otherwise the resolution the checkpoint was trained at."""
    h, w = size_hw
    saved = ckpt.get("args", {})
    h = h if h is not None else int(saved.get("height", 512))
    w = w if w is not None else int(saved.get("width", 256))
    return h, w


# ── Main ─────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate VISHC-PPD checkpoints.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset_root", type=str)
    src.add_argument("--test_dir", type=str, help="Folder of annotated jpg+json pairs.")
    p.add_argument("--split", default="test")
    p.add_argument("--runs_dir", type=str, help="runs/<model>/<run>/best.pt layout (multi-seed).")
    p.add_argument("--models", nargs="+", choices=MODEL_NAMES)
    p.add_argument("--checkpoints", nargs="+")
    p.add_argument("--out_dir", default="evaluation")
    p.add_argument("--height", type=int, default=None, help="Network input height (default: value stored in each checkpoint).")
    p.add_argument("--width", type=int, default=None, help="Network input width (default: value stored in each checkpoint).")
    p.add_argument("--latency_repeats", type=int, default=3)
    p.add_argument("--cpu", action="store_true")
    return p


def main(args: argparse.Namespace) -> None:
    if args.runs_dir:
        runs = discover_runs(Path(args.runs_dir))
    else:
        if not args.models or not args.checkpoints or len(args.models) != len(args.checkpoints):
            raise SystemExit("pass --runs_dir, or --models and --checkpoints with the same length")
        runs = [(m, Path(c).parent.name or "run", Path(c)) for m, c in zip(args.models, args.checkpoints)]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    samples = collect_from_dataset(args.dataset_root, args.split) if args.dataset_root else collect_from_folder(args.test_dir, require_json=True)
    samples = [s for s in samples if s.has_annotation]
    device = resolve_device(args.cpu)
    print(f"Device {device} | {len(samples)} annotated images | {len(runs)} checkpoint(s)\n")

    run_rows, per_image_rows = [], []
    confs: Dict[str, torch.Tensor] = {}
    by_model: Dict[str, List[dict]] = defaultdict(list)
    for model_name, run_name, ckpt in runs:
        print(f"Evaluating {model_name} [{run_name}] <- {ckpt}")
        mets, conf, per_img = evaluate_checkpoint(model_name, ckpt, samples, device, (args.height, args.width), args.latency_repeats)
        row = {"model": model_name, "run": run_name, "checkpoint": str(ckpt), **mets}
        run_rows.append(row)
        by_model[model_name].append(mets)
        confs[model_name] = confs.get(model_name, torch.zeros_like(conf)) + conf
        per_image_rows += [{"model": model_name, "run": run_name, **r} for r in per_img]
        print(f"  mIoU={mets['miou']:.4f}  F1={mets['f1']:.4f}  gAcc={mets['global_acc']:.4f}  "
              f"params={mets['params_M']:.1f}M  {mets['infer_ms']:.1f} ms/img")

    # ── aggregate ────────────────────────────────────────────────────────
    keys = [k for k in run_rows[0] if k not in ("model", "run", "checkpoint")]
    summary: Dict[str, dict] = {}
    for model_name, rows in by_model.items():
        summary[model_name] = {k: {"mean": float(np.mean([r[k] for r in rows])), "std": float(np.std([r[k] for r in rows])),
                                   "n": len(rows)} for k in keys}

    def write_csv(path: Path, rows: List[dict]):
        if not rows:
            return
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"  saved {path.name}")

    write_csv(out / "metrics_runs.csv", run_rows)
    write_csv(out / "metrics_per_image.csv", per_image_rows)
    summary_rows = []
    region_rows = []
    for m, s in summary.items():
        r = {"model": m, "display": DISPLAY_NAMES.get(m, m), "n_runs": s["miou"]["n"],
             "params_M": f"{s['params_M']['mean']:.1f}", "infer_ms": f"{s['infer_ms']['mean']:.1f}"}
        for k, _ in SCALARS:
            r[k] = f"{s[k]['mean']:.4f} ± {s[k]['std']:.4f}" if s[k]["n"] > 1 else f"{s[k]['mean']:.4f}"
        summary_rows.append(r)
        rr = {"model": m}
        for reg in REGION_LABELS:
            for kind in ("iou", "f1", "acc"):
                rr[f"{kind}_{reg}"] = f"{s[f'{kind}_{reg}']['mean']:.4f}"
        region_rows.append(rr)
    write_csv(out / "metrics_summary.csv", summary_rows)
    write_csv(out / "metrics_per_region.csv", region_rows)
    (out / "metrics_full.json").write_text(json.dumps({
        "runs": run_rows, "summary": summary, "n_test_images": len(samples),
        "confusion_summed_over_runs": {m: c.tolist() for m, c in confs.items()}}, indent=1), encoding="utf-8")

    print("\nSummary (test set):")
    for r in summary_rows:
        print(f"  {r['display']:18s} mIoU {r['miou']:>17s}  F1 {r['f1']:>17s}  gAcc {r['global_acc']:>17s}  "
              f"{r['params_M']:>6s}M  {r['infer_ms']:>6s} ms")

    print(f"\nDone -> {out}")


if __name__ == "__main__":
    main(build_parser().parse_args())
