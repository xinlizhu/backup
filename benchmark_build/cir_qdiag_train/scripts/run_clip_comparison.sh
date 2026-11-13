#!/usr/bin/env bash
set -euo pipefail

GPUS="${GPUS:-1}"
VARIANTS="${VARIANTS:-clip_b clip_l}"

for variant in ${VARIANTS}; do
  for mode in baseline add_substitute add_substitute_delete; do
    config="configs/${variant}_${mode}.yaml"
    output="outputs/${variant}/${mode}"
    torchrun --standalone --nproc_per_node="${GPUS}" scripts/train.py \
      --config "${config}"
    python scripts/evaluate.py \
      --config "${config}" \
      --checkpoint "${output}/last.pt"
  done

  python scripts/compare_training_modes.py \
    --baseline "outputs/${variant}/baseline/val_metrics.json" \
    --add-substitute "outputs/${variant}/add_substitute/val_metrics.json" \
    --add-substitute-delete "outputs/${variant}/add_substitute_delete/val_metrics.json" \
    --output "outputs/${variant}/comparison.json"
done
