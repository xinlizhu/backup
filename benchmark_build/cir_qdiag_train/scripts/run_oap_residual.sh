#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_ENV="${PYTHON_ENV:-/mnt/data/datasets/search_dataset/zxl/multi_modal_py3.12}"
GPU_ID="${GPU_ID:-0}"
MODELS="${MODELS:-clip_l,bge_vl_large}"

cd "$PROJECT_ROOT"
IFS=',' read -r -a MODEL_LIST <<< "$MODELS"

for MODEL_NAME in "${MODEL_LIST[@]}"; do
  case "$MODEL_NAME" in
    clip_l)
      CONFIG="configs/oap_residual_clip_l_beta0_1.yaml"
      OUTPUT_DIR="outputs/oap_residual/clip_l/beta0_1"
      ;;
    bge_vl_large)
      CONFIG="configs/oap_residual_bge_vl_large_beta0_1.yaml"
      OUTPUT_DIR="outputs/oap_residual/bge_vl_large/beta0_1"
      ;;
    *)
      echo "Unsupported model: $MODEL_NAME" >&2
      exit 2
      ;;
  esac

  mkdir -p "$OUTPUT_DIR"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_ENV/bin/torchrun" \
    --standalone \
    --nproc_per_node=1 \
    scripts/train_oap.py \
    --config "$CONFIG" \
    2>&1 | tee "$OUTPUT_DIR/train.log"

  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_ENV/bin/python" \
    scripts/evaluate_oap.py \
    --config "$CONFIG" \
    --checkpoint "$OUTPUT_DIR/last.pt" \
    --output "$OUTPUT_DIR/oap_cirr_metrics.json" \
    2>&1 | tee "$OUTPUT_DIR/evaluate.log"
done
