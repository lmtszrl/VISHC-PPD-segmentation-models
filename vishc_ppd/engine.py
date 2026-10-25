"""
Training / evaluation loops and checkpoint helpers shared by the CLI scripts.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .constants import NUM_CLASSES
from .data import build_eval_transform
from .metrics import make_empty_confusion, metrics_from_confusion, update_confusion
from .models import get_model

__all__ = [
    "resolve_device",
    "train_one_epoch",
    "evaluate_loader",
    "predict_image",
    "save_checkpoint",
    "load_checkpoint",
    "Predictor",
]


def resolve_device(force_cpu: bool = False) -> torch.device:
    if not force_cpu and torch.cuda.is_available():
        return torch.device("cuda")
    if not force_cpu and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled and device.type == "cuda")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
    aux_weight: float = 0.4,
    amp: bool = True,
    grad_clip: Optional[float] = 1.0,
    desc: str = "train",
) -> float:
    model.train()
    total, n = 0.0, 0
    for x, y, _ in tqdm(loader, desc=f"  {desc}", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp):
            out = model(x)
            loss = loss_fn(out["out"], y)
            if out.get("aux") is not None and aux_weight > 0:
                loss = loss + aux_weight * loss_fn(out["aux"], y)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        total += float(loss.item())
        n += 1
    return total / max(1, n)


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: Optional[nn.Module] = None,
    amp: bool = True,
    desc: str = "val",
) -> Tuple[Optional[float], Dict[str, float]]:
    """Metrics at the network resolution (fast; used for model selection during training)."""
    model.eval()
    conf = make_empty_confusion(NUM_CLASSES, device=device)
    losses: List[float] = []
    for x, y, _ in tqdm(loader, desc=f"  {desc}", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with _autocast(device, amp):
            logits = model(x)["out"]
        if loss_fn is not None:
            losses.append(float(loss_fn(logits.float(), y).item()))
        update_confusion(conf, logits.argmax(1), y)
    mets = metrics_from_confusion(conf)
    return (float(np.mean(losses)) if losses else None), mets


# ── Checkpoints ──────────────────────────────────────────────────────────────

def save_checkpoint(path: Path | str, model: nn.Module, model_name: str, epoch: int,
                    args: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> None:
    payload = {"model_name": model_name, "epoch": epoch, "model_state": model.state_dict(), "args": args}
    if extra:
        payload.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path | str, model_name: Optional[str] = None, device: torch.device = torch.device("cpu"),
                    **model_kwargs) -> Tuple[nn.Module, Dict[str, Any]]:
    """Rebuild the model recorded in a checkpoint and load its weights (no hub download)."""
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    name = model_name or ckpt.get("model_name")
    if name is None:
        raise ValueError("checkpoint does not record the model name; pass model_name explicitly")
    saved = ckpt.get("args", {})
    if "input_hw" not in model_kwargs and "height" in saved and "width" in saved:
        model_kwargs["input_hw"] = (int(saved["height"]), int(saved["width"]))
    model = get_model(name, num_classes=NUM_CLASSES, pretrained=False, **model_kwargs)
    model.load_state_dict(ckpt["model_state"], strict=True)
    return model.to(device).eval(), ckpt


# ── Single-image inference ───────────────────────────────────────────────────

class Predictor:
    """Runs a model on full-resolution RGB images and returns label masks at the original size."""

    def __init__(self, model: nn.Module, device: torch.device, size_hw: Tuple[int, int], amp: bool = True):
        self.model = model.to(device).eval()
        self.device = device
        self.size_hw = tuple(size_hw)
        self.tf = build_eval_transform(self.size_hw)
        self.amp = amp

    @torch.no_grad()
    def logits(self, img_rgb: np.ndarray) -> torch.Tensor:
        x = self.tf(image=img_rgb)["image"].unsqueeze(0).to(self.device)
        with _autocast(self.device, self.amp):
            return self.model(x)["out"].float()

    @torch.no_grad()
    def __call__(self, img_rgb: np.ndarray) -> np.ndarray:
        h0, w0 = img_rgb.shape[:2]
        pred = self.logits(img_rgb).argmax(1)[0].cpu().numpy().astype(np.uint8)
        return cv2.resize(pred, (w0, h0), interpolation=cv2.INTER_NEAREST)

    @torch.no_grad()
    def timed(self, img_rgb: np.ndarray, repeats: int = 1) -> Tuple[np.ndarray, float]:
        """Prediction plus mean forward-pass time in milliseconds (network resolution)."""
        x = self.tf(image=img_rgb)["image"].unsqueeze(0).to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            with _autocast(self.device, self.amp):
                logits = self.model(x)["out"].float()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0 / repeats
        h0, w0 = img_rgb.shape[:2]
        pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
        return cv2.resize(pred, (w0, h0), interpolation=cv2.INTER_NEAREST), ms


def predict_image(model: nn.Module, img_rgb: np.ndarray, device: torch.device, size_hw: Tuple[int, int]) -> np.ndarray:
    return Predictor(model, device, size_hw)(img_rgb)
