#!/usr/bin/env bash
set -euo pipefail

config="configs/bge_vl_large_full_hard_w0_1.yaml"
output="outputs/bge_vl_large/full/hard_w0_1"
final_checkpoint="${output}/checkpoint_epoch_010.pt"

if [[ -f "${final_checkpoint}" ]]; then
  echo "[skip train] completed checkpoint: ${final_checkpoint}"
elif [[ -f "${output}/last.pt" ]]; then
  echo "[resume train] ${output}/last.pt"
  torchrun --standalone --nproc_per_node=1 scripts/train.py \
    --config "${config}" \
    --resume-from "${output}/last.pt"
else
  torchrun --standalone --nproc_per_node=1 scripts/train.py \
    --config "${config}"
fi

python scripts/evaluate.py \
  --config "${config}" \
  --checkpoint "${final_checkpoint}"
