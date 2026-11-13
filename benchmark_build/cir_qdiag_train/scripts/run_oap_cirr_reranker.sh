#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_ENV="${PYTHON_ENV:-/mnt/data/datasets/search_dataset/zxl/multi_modal_py3.12}"
GPU_ID="${GPU_ID:-0}"
TOP_KS="${TOP_KS:-10,50,100}"
OUTPUT_ROOT="outputs/oap_residual/cirr_reranker"

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_ROOT"

run_one() {
  local name="$1"
  local config="$2"
  local checkpoint="$3"
  local betas="$4"
  local output="$OUTPUT_ROOT/$name.json"
  if [[ -s "$output" ]]; then
    echo "Already evaluated: $output"
    return
  fi
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_ENV/bin/python" \
    scripts/evaluate_oap_reranker.py \
    --config "$config" \
    --checkpoint "$checkpoint" \
    --betas "$betas" \
    --top-ks "$TOP_KS" \
    --output "$output"
}

run_one \
  "clip_l_epoch_002" \
  "configs/oap_residual_clip_l_beta0_1.yaml" \
  "outputs/oap_residual/clip_l/beta0_1/checkpoint_epoch_002.pt" \
  "0.05,0.1,0.2"

run_one \
  "clip_l_epoch_005" \
  "configs/oap_residual_clip_l_beta0_1.yaml" \
  "outputs/oap_residual/clip_l/beta0_1/checkpoint_epoch_005.pt" \
  "0.05,0.1,0.2"

run_one \
  "bge_vl_large_epoch_005" \
  "configs/oap_residual_bge_vl_large_beta0_1.yaml" \
  "outputs/oap_residual/bge_vl_large/beta0_1/checkpoint_epoch_005.pt" \
  "0.025,0.1,0.2"

"$PYTHON_ENV/bin/python" scripts/summarize_oap_reranker.py \
  --root "$OUTPUT_ROOT"
