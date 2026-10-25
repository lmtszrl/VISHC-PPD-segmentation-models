#!/usr/bin/env python3
"""
Run inference with a trained checkpoint on a folder of images (or on the test
split of the released dataset) and export the results.

Examples
--------
python predict.py --checkpoint runs/unet/seed42/best.pt \
    --dataset_root "/path/to/VISHC-PPD-dataset" --split test \
    --out_dir preds/unet --save_json

python predict.py --checkpoint runs/unet/seed42/best.pt --image_dir /path/to/new_images --out_dir preds/new

Outputs
-------
<out_dir>/<image_id>_mask.png   label mask (0 = background, 1..8 = P1..P8) as an 8-bit PNG
<out_dir>/<image_id>.json       LabelMe-compatible polygons (--save_json)
<out_dir>/region_pixels.csv     pixel count per region for every image
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vishc_ppd.data import collect_from_dataset, collect_from_folder, read_image_rgb, write_labelme_json  # noqa: E402
from vishc_ppd.engine import Predictor, load_checkpoint, resolve_device  # noqa: E402
from vishc_ppd.models import MODEL_NAMES  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="VISHC-PPD inference.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", choices=MODEL_NAMES, default=None, help="Override the model name stored in the checkpoint.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset_root", type=str)
    src.add_argument("--image_dir", type=str, help="Any folder of *.jpg / *.png images.")
    p.add_argument("--split", default="test", choices=["train", "test", "all"], help="Used with --dataset_root.")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--height", type=int, default=None, help="Default: value stored in the checkpoint.")
    p.add_argument("--width", type=int, default=None, help="Default: value stored in the checkpoint.")
    p.add_argument("--save_json", action="store_true", help="Also export LabelMe-compatible JSON polygons.")
    p.add_argument("--cpu", action="store_true")
    return p


def main(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = collect_from_dataset(args.dataset_root, args.split) if args.dataset_root else collect_from_folder(args.image_dir)
    device = resolve_device(args.cpu)
    model, ckpt = load_checkpoint(args.checkpoint, args.model, device)
    print(f"Model  : {ckpt.get('model_name')} (epoch {ckpt.get('epoch')})  | device: {device}")
    print(f"Images : {len(samples)}")

    saved = ckpt.get("args", {})
    size_hw = (args.height or int(saved.get("height", 512)), args.width or int(saved.get("width", 256)))
    predictor = Predictor(model, device, size_hw)
    rows = []
    for s in tqdm(samples, desc="predict"):
        img = read_image_rgb(s.image_path)
        pred = predictor(img)
        Image.fromarray(pred).save(out_dir / f"{s.image_id}_mask.png")
        if args.save_json:
            write_labelme_json(out_dir / f"{s.image_id}.json", s.image_path.name, pred)
        rows.append([s.image_id] + [int((pred == k).sum()) for k in range(1, 9)])

    with (out_dir / "region_pixels.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id"] + [f"P{k}_pixels" for k in range(1, 9)])
        w.writerows(rows)
    print(f"Done -> {out_dir}")


if __name__ == "__main__":
    main(build_parser().parse_args())
