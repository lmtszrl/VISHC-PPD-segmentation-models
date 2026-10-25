# VISHC-PPD-segmentation-models

Reference code for the deep-learning benchmark reported in

> *A Vietnamese child and adult plantar pressure image dataset with eight-region annotations for foot pathology* (Data Descriptor, in preparation; corresponding author Hieu H. Pham, VinUniversity).

The dataset (536 plantar-pressure images from 196 subjects, LabelMe polygons for the eight plantar regions P1–P8) is available on figshare under the DOI [10.6084/m9.figshare.34781376](https://doi.org/10.6084/m9.figshare.34781376) (CC BY 4.0) and is described in the README shipped with the data. This repository trains and evaluates five segmentation models under one protocol:

| model id | architecture | encoder pre-training | extra dependency |
|---|---|---|---|
| `deeplabv3_r50` | DeepLabV3 + FCN aux head, ResNet-50 | COCO/VOC (torchvision) | – |
| `unet` | U-Net, ResNet-34 encoder | ImageNet-1k (torchvision) | – |
| `segformer_b2` | SegFormer-B2 (MiT-B2 + MLP decoder) | ImageNet-1k (`nvidia/mit-b2`) | `transformers` |
| `swin_unet` | Swin-Tiny encoder + conv U-Net decoder | ImageNet-1k (timm) | `timm` |
| `dinov2_vitb14` | DINOv2 ViT-B/14 encoder + conv decoder | LVD-142M self-supervised (Meta release, timm) | `timm` |

All models share one interface, `model(x)["out"] → (N, 9, H, W)` logits (class 0 = background, 1–8 = P1–P8).

## 1. Layout

```
VISHC-PPD-segmentation-models/
├── train.py               train one model on the training split (subject-level validation hold-out)
├── predict.py             inference → label masks (PNG) and LabelMe JSON
├── evaluate.py            test-split metrics (native resolution, background-aware), multi-seed mean ± SD
├── validate_dataset.py    checks a dataset copy against the documented contract
├── scripts/run_benchmark.sh   all models × 3 seeds + evaluation
├── scripts/make_reference_split.py   generates the released subject-level train/test split from the metadata table
├── scripts/annotation_consistency.py   region-area, paired-agreement, protocol-conformance and overlap statistics of the polygons
├── notebooks/VISHC_PPD_Colab.ipynb   the same pipeline on Google Colab
├── vishc_ppd/             the package
│   ├── constants.py       class ids, names, colours, naming pattern
│   ├── data/              annotations.py (LabelMe I/O + rasterisation), splits.py, dataset.py
│   ├── models/            one file per architecture + registry (get_model)
│   ├── losses.py          cross-entropy + soft Dice
│   ├── metrics.py         confusion-matrix metrics
│   ├── engine.py          training / evaluation loops, checkpoints, Predictor
│   └── utils/seed.py
└── tests/                 pytest suite (+ 4 sample image/JSON pairs)
```

## 2. Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest            # optional: 20 unit tests, CPU only, ~15 s
```

Python ≥ 3.10, PyTorch ≥ 2.2. A CUDA GPU with ≥ 12 GB is recommended for training (batch 8 at 512 × 256); CPU works for inference and tests.

**Pretrained weights.** All encoder weights are public and are downloaded automatically on first use: DeepLabV3 and ResNet-34 from the torchvision servers, Swin-Tiny from the official Swin Transformer GitHub release, DINOv2 ViT-B/14 from Meta's release server (both through timm), and MiT-B2 (SegFormer) from the `transformers` model repository. No account or access token is needed for any of them.

## 3. Data conventions implemented here

* **Inputs.** The released dataset root (`VISHC-PPD-dataset/`) with `Full data (unclassified)/`, `train_test_split.csv` and `VISHC-PPD_metadata.csv`. Any plain folder of `<stem>.jpg` + `<stem>.json` pairs is also accepted (`--train_dir`, `--test_dir`, `--image_dir`).
* **Rasterisation rule** (`vishc_ppd/data/annotations.py`). Polygons are painted in descending order of area, so where two regions overlap the smaller one wins — independent of the order of shapes in the JSON. Vertices are clipped to the image (35 polygons have vertices up to 1.3 px outside it), and self-intersecting outlines (50 polygons in the release) are repaired with `shapely.make_valid` before filling. Overlaps between neighbouring polygons are small (median 5 px² per pair, at most 1.05 % of a footprint per image; `scripts/annotation_consistency.py` reports these numbers). Regions that are absent from a footprint (5 documented images) simply produce no pixels.
* **Labels** are parsed strictly (`P1`…`P8` only); anything else raises an error, so a corrupted annotation cannot silently become background.
* **Validation split** inside `train.py` is made by subject (`S0001` …), never by image, so the two feet / two devices of one participant never straddle train and validation.
* **Metrics** (`vishc_ppd/metrics.py`) are computed at the native image resolution from a 9 × 9 confusion matrix that includes background; scores are then reported for P1–P8. False positives painted onto the background therefore lower IoU and precision.
* **No flips** in augmentation: left/right mirroring would swap the medial and lateral regions.

## 4. Usage

```bash
DATA="/path/to/VISHC-PPD-dataset"

# 0) sanity-check the data copy
python validate_dataset.py --dataset_root "$DATA"

# 1) train (defaults: 60 epochs, batch 8, AdamW, head LR 3e-4 / encoder LR 3e-5, 3 warm-up epochs, cosine, 512x256,
#    encoder frozen for the first 3 epochs — the same protocol for every model)
python train.py --model unet          --dataset_root "$DATA" --out_dir runs/unet/seed42          --seed 42
python train.py --model dinov2_vitb14 --dataset_root "$DATA" --out_dir runs/dinov2_vitb14/seed42 --seed 42

# 2) evaluate every runs/<model>/<seed>/best.pt on the test split (mean ± SD over seeds)
python evaluate.py --dataset_root "$DATA" --runs_dir runs --out_dir evaluation

# 3) export predictions
python predict.py --checkpoint runs/unet/seed42/best.pt --dataset_root "$DATA" --split test --out_dir preds/unet --save_json
python predict.py --checkpoint runs/unet/seed42/best.pt --image_dir /path/to/new_images --out_dir preds/new

# full benchmark (5 models × 3 seeds) in one go
bash scripts/run_benchmark.sh "$DATA"
```

Useful `train.py` options: `--backbone_lr_mult` (encoder LR = mult × `--lr`; default 0.1 for all models), `--warmup_epochs` (default 3), `--freeze_backbone_epochs 0`, `--val_fraction`, `--no_amp`, `--cpu`. Every run writes `args.json`, `split.json` (exact validation image ids), `metrics.json` (per epoch) and `best.pt` / `last.pt` (state dict + the arguments needed to rebuild the model).

`evaluate.py` writes `metrics_summary.csv` (one row per model, mean ± SD, parameter count, inference time), `metrics_runs.csv` (one row per checkpoint), `metrics_per_region.csv`, `metrics_per_image.csv` (per-image mIoU for statistical tests) and `metrics_full.json` (incl. confusion matrices).

## 5. Benchmark results

Test split (108 images), native resolution, mean ± SD over seeds 42/43/44 (population SD), trained with the default hyper-parameters on one NVIDIA A100 (`scripts/run_benchmark.sh` or the Colab notebook). Inference time is the forward pass per image at 512 × 256, batch size 1, mixed precision.

| model id | params (M) | time (ms) | mIoU | F1 |
|---|---:|---:|---|---|
| `deeplabv3_r50` | 42.0 | 10.2 | 0.7869 ± 0.0011 | 0.8782 ± 0.0007 |
| `dinov2_vitb14` | 88.6 | 11.6 | 0.7863 ± 0.0049 | 0.8777 ± 0.0032 |
| `segformer_b2` | 27.4 | 25.6 | 0.7830 ± 0.0005 | 0.8755 ± 0.0004 |
| `swin_unet` | 31.6 | 18.2 | 0.7755 ± 0.0016 | 0.8703 ± 0.0011 |
| `unet` | 24.5 | 8.6 | 0.7741 ± 0.0018 | 0.8695 ± 0.0011 |

Per-region IoU, per-image scores and the full tables are produced by `evaluate.py`. Bit-exact agreement across GPUs is not guaranteed by CUDA; small run-to-run differences in the mean metrics are expected.

## 6. Adding a model

Subclass `vishc_ppd.models.base.SegmentationModel`, return `{"out": logits}` at input resolution, implement `backbone_parameters()`, and register the class in `vishc_ppd/models/__init__.py`. Everything else (training, evaluation, export) works unchanged.

## 7. Citation and licence

Code: MIT licence (`LICENSE`). Please cite the dataset by its DOI (10.6084/m9.figshare.34781376) and, once published, the Data Descriptor above; `CITATION.cff` carries the metadata for citation managers and GitHub's *Cite this repository* button. Data: CC BY 4.0, see the dataset README.
