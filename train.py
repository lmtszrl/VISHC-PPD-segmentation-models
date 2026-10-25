#!/usr/bin/env python3
"""
Train one segmentation model on the VISHC-PPD training split.

Examples
--------
# From the released dataset root (recommended; uses train_test_split.csv)
python train.py --model unet --dataset_root "/path/to/VISHC-PPD-dataset" --out_dir runs/unet/seed42

# From a plain folder of jpg+json pairs
python train.py --model unet --train_dir "/path/to/train data" --out_dir runs/unet/seed42

# DINOv2 ViT-B/14 encoder + convolutional decoder
python train.py --model dinov2_vitb14 --dataset_root ... --out_dir runs/dinov2_vitb14/seed42

Validation images are held out *by subject*, so both feet / both devices of a
participant are always on the same side of the split.

Outputs (in --out_dir): best.pt, last.pt, metrics.json (per epoch), args.json,
split.json (image ids used for training / validation).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vishc_ppd import __version__  # noqa: E402
from vishc_ppd.constants import NUM_CLASSES  # noqa: E402
from vishc_ppd.data import PlantarDataset, collect_from_dataset, collect_from_folder, subject_level_split, summarize  # noqa: E402
from vishc_ppd.engine import evaluate_loader, resolve_device, save_checkpoint, train_one_epoch  # noqa: E402
from vishc_ppd.losses import CompoundLoss  # noqa: E402
from vishc_ppd.models import MODEL_NAMES, get_model  # noqa: E402
from vishc_ppd.utils.seed import seed_everything  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a VISHC-PPD segmentation model.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", required=True, choices=MODEL_NAMES)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset_root", type=str, help="Released dataset folder (uses train_test_split.csv).")
    src.add_argument("--train_dir", type=str, help="Folder with paired *.jpg + *.json files.")
    p.add_argument("--out_dir", required=True, type=str)

    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--backbone_lr_mult", type=float, default=0.1,
                   help="Learning-rate multiplier for the pretrained encoder (head trains at --lr, encoder at lr × mult). "
                        "Applied uniformly to every model in the benchmark.")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=3, help="Linear LR warm-up epochs before cosine decay.")
    p.add_argument("--freeze_backbone_epochs", type=int, default=3,
                   help="Freeze the backbone for the first N epochs (0 = train everything from epoch 1).")
    p.add_argument("--val_fraction", type=float, default=0.10, help="Fraction of training images held out (by subject).")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--dice_weight", type=float, default=1.0)
    p.add_argument("--aux_weight", type=float, default=0.4, help="Auxiliary-head loss weight (DeepLabV3 only).")
    p.add_argument("--grad_clip", type=float, default=1.0, help="Gradient-norm clipping (0 = off).")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_pretrained", action="store_true", help="Random initialisation (for tests / ablation).")
    p.add_argument("--no_amp", action="store_true", help="Disable mixed precision on CUDA.")
    p.add_argument("--cpu", action="store_true")
    return p


def make_optimizer(model, args) -> torch.optim.Optimizer:
    bb_ids = {id(p) for p in model.backbone_parameters()}
    bb = [p for p in model.parameters() if id(p) in bb_ids and p.requires_grad]
    head = [p for p in model.parameters() if id(p) not in bb_ids and p.requires_grad]
    groups = [{"params": head, "lr": args.lr}]
    if bb:
        groups.append({"params": bb, "lr": args.lr * args.backbone_lr_mult})
    return torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)


def make_scheduler(optimizer, args, total_epochs: int, completed_epochs: int = 0):
    """Linear warm-up then cosine decay to 1 % of the base LR; resumable at ``completed_epochs``."""
    warm = max(0, args.warmup_epochs)

    def lr_lambda(epoch: int) -> float:  # epoch counts completed epochs (0-based)
        if epoch < warm:
            return (epoch + 1) / (warm + 1)
        t = (epoch - warm) / max(1, total_epochs - warm)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    if completed_epochs > 0:
        for g in optimizer.param_groups:
            g.setdefault("initial_lr", g["lr"])
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=completed_epochs - 1)


def main(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # ── data ─────────────────────────────────────────────────────────────
    if args.dataset_root:
        all_samples = collect_from_dataset(args.dataset_root, "train")
    else:
        all_samples = collect_from_folder(args.train_dir, require_json=True)
    # code version and a fingerprint of the training pool (sorted image ids) are stored with every run,
    # so that a checkpoint can be matched to the data it was trained on
    pool_fingerprint = hashlib.sha1("\n".join(sorted(s.image_id for s in all_samples)).encode()).hexdigest()[:12]
    run_args = {**vars(args), "package_version": __version__, "train_pool": f"{len(all_samples)}:{pool_fingerprint}"}
    (out_dir / "args.json").write_text(json.dumps(run_args, indent=2), encoding="utf-8")
    tr_samples, va_samples = subject_level_split(all_samples, args.val_fraction, args.seed)
    print(f"Training pool : {summarize(all_samples)}")
    print(f"  train       : {summarize(tr_samples)}")
    print(f"  val         : {summarize(va_samples)}  (subject-disjoint)")
    (out_dir / "split.json").write_text(json.dumps({
        "train": [s.image_id for s in tr_samples], "val": [s.image_id for s in va_samples]}, indent=1), encoding="utf-8")

    device = resolve_device(args.cpu)
    size_hw = (args.height, args.width)
    tr_loader = DataLoader(PlantarDataset(tr_samples, size_hw, train=True), batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers, pin_memory=(device.type == "cuda"), drop_last=True)
    va_loader = DataLoader(PlantarDataset(va_samples, size_hw, train=False), batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    if len(tr_loader) == 0:
        raise SystemExit("batch_size is larger than the training set; reduce --batch_size")

    # ── model / optimisation ────────────────────────────────────────────
    amp = not args.no_amp
    model = get_model(args.model, num_classes=NUM_CLASSES, pretrained=not args.no_pretrained, input_hw=size_hw).to(device)
    print(f"Model : {args.model} ({model.display_name}), {model.count_parameters() / 1e6:.1f} M params")
    print(f"Device: {device}  | AMP: {amp and device.type == 'cuda'}")

    frozen = args.freeze_backbone_epochs > 0
    if frozen:
        for p in model.backbone_parameters():
            p.requires_grad = False
        print(f"Backbone frozen for the first {args.freeze_backbone_epochs} epoch(s).")

    optimizer = make_optimizer(model, args)
    scheduler = make_scheduler(optimizer, args, args.epochs)
    loss_fn = CompoundLoss(NUM_CLASSES, ignore_index=0, dice_weight=args.dice_weight)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp and device.type == "cuda"))

    best_miou, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        if frozen and epoch == args.freeze_backbone_epochs + 1:
            for p in model.backbone_parameters():
                p.requires_grad = True
            # Rebuild optimiser/scheduler so the backbone joins with its own LR;
            # the cosine schedule continues from the current epoch.
            optimizer = make_optimizer(model, args)
            scheduler = make_scheduler(optimizer, args, args.epochs, completed_epochs=epoch - 1)
            frozen = False
            print("  Backbone unfrozen.")

        tr_loss = train_one_epoch(model, tr_loader, optimizer, loss_fn, device, scaler,
                                  aux_weight=args.aux_weight, amp=amp, grad_clip=args.grad_clip or None)
        va_loss, mets = evaluate_loader(model, va_loader, device, loss_fn, amp=amp)
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:03d}/{args.epochs} | lr={lr_now:.2e} | train={tr_loss:.4f} | val={va_loss:.4f} | "
              f"mIoU={mets['miou']:.4f} | F1={mets['f1']:.4f} | gAcc={mets['global_acc']:.4f}")
        history.append({"epoch": epoch, "lr": lr_now, "train_loss": tr_loss, "val_loss": va_loss, **mets})
        (out_dir / "metrics.json").write_text(json.dumps(history, indent=1), encoding="utf-8")

        save_checkpoint(out_dir / "last.pt", model, args.model, epoch, run_args)
        if mets["miou"] > best_miou:
            best_miou = mets["miou"]
            save_checkpoint(out_dir / "best.pt", model, args.model, epoch, run_args, {"best_val_miou": best_miou})
            print(f"  * new best val mIoU {best_miou:.4f} -> best.pt")

    print(f"\nDone. Best validation mIoU = {best_miou:.4f}. Checkpoints in {out_dir}")


if __name__ == "__main__":
    main(build_parser().parse_args())
