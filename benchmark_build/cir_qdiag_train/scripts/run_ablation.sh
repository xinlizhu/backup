#!/usr/bin/env bash
set -euo pipefail

GPUS="${GPUS:-1}"

torchrun --standalone --nproc_per_node="${GPUS}" scripts/train.py \
  --config configs/clip_b_add_substitute_delete.yaml

torchrun --standalone --nproc_per_node="${GPUS}" scripts/train.py \
  --config configs/clip_b_add_substitute.yaml

python scripts/evaluate.py \
  --config configs/clip_b_add_substitute_delete.yaml \
  --checkpoint outputs/clip_b/add_substitute_delete/last.pt

python scripts/evaluate.py \
  --config configs/clip_b_add_substitute.yaml \
  --checkpoint outputs/clip_b/add_substitute/last.pt

python scripts/compare_ablation.py \
  --with-delete outputs/clip_b/add_substitute_delete/val_metrics.json \
  --without-delete outputs/clip_b/add_substitute/val_metrics.json
