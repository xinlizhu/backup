#!/usr/bin/env bash
set -euo pipefail

# Full CLIP-L hard-negative weight sweep. The completed 0.0 baseline and 0.5
# run are reused by the final summary. Override WEIGHTS to train a subset.
WEIGHTS="${WEIGHTS:-0.1 0.25 0.75 1.0}"

weight_tag() {
  case "$1" in
    0.1) echo "0_1" ;;
    0.25) echo "0_25" ;;
    0.75) echo "0_75" ;;
    1.0) echo "1_0" ;;
    *)
      echo "Unsupported weight: $1; expected 0.1, 0.25, 0.75, or 1.0" >&2
      return 2
      ;;
  esac
}

summary_weights="0.0,0.5"
for weight in ${WEIGHTS}; do
  tag="$(weight_tag "${weight}")"
  config="configs/clip_l_full_hard_w${tag}.yaml"
  output="outputs/clip_l/full/hard_w${tag}"
  final_checkpoint="${output}/checkpoint_epoch_010.pt"

  if [[ -f "${final_checkpoint}" ]]; then
    echo "[skip train] completed checkpoint: ${final_checkpoint}"
  elif [[ -f "${output}/last.pt" ]]; then
    echo "[resume train] ${output}/last.pt"
    torchrun --standalone --nproc_per_node=1 scripts/train.py \
      --config "${config}" \
      --resume-from "${output}/last.pt"
  else
    echo "[train] hard_negative_weight=${weight}"
    torchrun --standalone --nproc_per_node=1 scripts/train.py \
      --config "${config}"
  fi

  python scripts/evaluate.py \
    --config "${config}" \
    --checkpoint "${final_checkpoint}"
  summary_weights="${summary_weights},${weight}"
done

python scripts/summarize_full_hard_grid.py \
  --root outputs/clip_l/full \
  --weights "${summary_weights}" \
  --output outputs/clip_l/full/hard_weight_grid_cirr.csv
