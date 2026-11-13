#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/reverse_search/clip_l_full}"
FORCE="${FORCE:-0}"

run_one() {
  local model="$1"
  local config="$2"
  local checkpoint="$3"
  local output="${OUTPUT_ROOT}/${model}.json"

  if [[ -f "${output}" && "${FORCE}" != "1" ]]; then
    echo "[cache] ${output}"
    return
  fi
  echo "[reverse evaluate] ${model}"
  "${PYTHON}" scripts/evaluate_reverse.py \
    --config "${config}" \
    --checkpoint "${checkpoint}" \
    --output "${output}"
}

run_one \
  baseline_w0 \
  configs/clip_l_full_baseline.yaml \
  outputs/clip_l/full/baseline/checkpoint_epoch_010.pt
run_one \
  hard_w0_1 \
  configs/clip_l_full_hard_w0_1.yaml \
  outputs/clip_l/full/hard_w0_1/checkpoint_epoch_010.pt
run_one \
  hard_w0_25 \
  configs/clip_l_full_hard_w0_25.yaml \
  outputs/clip_l/full/hard_w0_25/checkpoint_epoch_010.pt

"${PYTHON}" scripts/summarize_reverse_search.py \
  --input "baseline_w0=${OUTPUT_ROOT}/baseline_w0.json" \
  --input "hard_w0_1=${OUTPUT_ROOT}/hard_w0_1.json" \
  --input "hard_w0_25=${OUTPUT_ROOT}/hard_w0_25.json" \
  --output "${OUTPUT_ROOT}/comparison.csv"
