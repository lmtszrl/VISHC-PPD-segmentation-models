#!/usr/bin/env bash
# Train the five benchmark models with three seeds each, then evaluate on the test split.
#
#   bash scripts/run_benchmark.sh "/path/to/VISHC-PPD-dataset" [runs_dir] [evaluation_dir]
#
# Pretrained encoder weights are downloaded automatically on first use (torchvision, timm, transformers).
set -euo pipefail
DATASET_ROOT="${1:?dataset root required}"
RUNS_DIR="${2:-runs}"
EVAL_DIR="${3:-evaluation}"
MODELS=(deeplabv3_r50 unet segformer_b2 swin_unet dinov2_vitb14)
SEEDS=(42 43 44)

for MODEL in "${MODELS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    OUT="${RUNS_DIR}/${MODEL}/seed${SEED}"
    if [[ -f "${OUT}/best.pt" ]]; then echo "skip ${OUT} (exists)"; continue; fi
    python train.py --model "${MODEL}" --dataset_root "${DATASET_ROOT}" --out_dir "${OUT}" --seed "${SEED}"
  done
done

python evaluate.py --dataset_root "${DATASET_ROOT}" --runs_dir "${RUNS_DIR}" --out_dir "${EVAL_DIR}"
