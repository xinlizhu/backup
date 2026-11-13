#!/usr/bin/env bash
set -euo pipefail

# Controlled partial-finetuning comparison. Both runs update the last four
# vision/text Transformer blocks, both CLIP projections, and the fusion head.
for mode in baseline hard_w0_5; do
  config="configs/clip_l_last4_${mode}.yaml"
  output="outputs/clip_l/last4/${mode}"
  torchrun --standalone --nproc_per_node=1 scripts/train.py \
    --config "${config}"
  python scripts/evaluate.py \
    --config "${config}" \
    --checkpoint "${output}/last.pt"
done

python scripts/compare_pair.py \
  --baseline outputs/clip_l/last4/baseline/val_metrics.json \
  --hard outputs/clip_l/last4/hard_w0_5/val_metrics.json \
  --output outputs/clip_l/last4/comparison.json
