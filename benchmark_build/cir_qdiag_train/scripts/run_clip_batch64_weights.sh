#!/usr/bin/env bash
set -euo pipefail

# One GPU, CLIP-L only by default. Override VARIANTS to run another subset.
# Every experiment is single-GPU, preserving global InfoNCE batch=64.
VARIANTS="${VARIANTS:-clip_l}"

for variant in ${VARIANTS}; do
  if [[ "${variant}" != "clip_b" && "${variant}" != "clip_l" ]]; then
    echo "Unsupported variant: ${variant}; expected clip_b or clip_l" >&2
    exit 2
  fi
  for mode in baseline hard_w0_5 hard_w1_0; do
    config="configs/${variant}_b64_${mode}.yaml"
    output="outputs/${variant}/batch64/${mode}"
    torchrun --standalone --nproc_per_node=1 scripts/train.py \
      --config "${config}"
    python scripts/evaluate.py \
      --config "${config}" \
      --checkpoint "${output}/last.pt"
  done

  python scripts/compare_hard_weights.py \
    --baseline "outputs/${variant}/batch64/baseline/val_metrics.json" \
    --weight-0-5 "outputs/${variant}/batch64/hard_w0_5/val_metrics.json" \
    --weight-1-0 "outputs/${variant}/batch64/hard_w1_0/val_metrics.json" \
    --output "outputs/${variant}/batch64/comparison.json"
done
