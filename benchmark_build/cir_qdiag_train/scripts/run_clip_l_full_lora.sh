#!/usr/bin/env bash
set -euo pipefail

# Four controlled CLIP-L runs by default. Set METHODS="full" or
# METHODS="lora_r16" to run one pair only.
METHODS="${METHODS:-lora_r16 full}"

for method in ${METHODS}; do
  if [[ "${method}" != "full" && "${method}" != "lora_r16" ]]; then
    echo "Unsupported method: ${method}; expected full or lora_r16" >&2
    exit 2
  fi
  for mode in baseline hard_w0_5; do
    config="configs/clip_l_${method}_${mode}.yaml"
    output="outputs/clip_l/${method}/${mode}"
    torchrun --standalone --nproc_per_node=1 scripts/train.py \
      --config "${config}"
    python scripts/evaluate.py \
      --config "${config}" \
      --checkpoint "${output}/last.pt"
  done
  python scripts/compare_pair.py \
    --baseline "outputs/clip_l/${method}/baseline/val_metrics.json" \
    --hard "outputs/clip_l/${method}/hard_w0_5/val_metrics.json" \
    --output "outputs/clip_l/${method}/comparison.json"
done
