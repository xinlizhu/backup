#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_ENV="${PYTHON_ENV:-/mnt/data/datasets/search_dataset/zxl/multi_modal_py3.12}"
GPU_ID="${GPU_ID:-0}"
MODELS="${MODELS:-clip_l,bge_vl_large}"
BETAS="${BETAS:-0.025,0.05,0.1,0.2}"
EPOCHS="${EPOCHS:-002,004,005}"

cd "$PROJECT_ROOT"
IFS=',' read -r -a MODEL_LIST <<< "$MODELS"
IFS=',' read -r -a BETA_LIST <<< "$BETAS"
IFS=',' read -r -a EPOCH_LIST <<< "$EPOCHS"

for MODEL_NAME in "${MODEL_LIST[@]}"; do
  case "$MODEL_NAME" in
    clip_l)
      CONFIG="configs/oap_residual_clip_l_beta0_1.yaml"
      RUN_DIR="outputs/oap_residual/clip_l/beta0_1"
      ;;
    bge_vl_large)
      CONFIG="configs/oap_residual_bge_vl_large_beta0_1.yaml"
      RUN_DIR="outputs/oap_residual/bge_vl_large/beta0_1"
      ;;
    *)
      echo "Unsupported model: $MODEL_NAME" >&2
      exit 2
      ;;
  esac

  SWEEP_DIR="$RUN_DIR/eval_sweep"
  mkdir -p "$SWEEP_DIR"

  # First compare the saved training epochs at the trained beta=0.1.
  for EPOCH in "${EPOCH_LIST[@]}"; do
    CHECKPOINT="$RUN_DIR/checkpoint_epoch_${EPOCH}.pt"
    OUTPUT="$SWEEP_DIR/epoch_${EPOCH}_beta_0.1.json"
    if [[ ! -f "$CHECKPOINT" ]]; then
      echo "Missing checkpoint, skipping: $CHECKPOINT" >&2
      continue
    fi
    if [[ -s "$OUTPUT" ]]; then
      echo "Already evaluated: $OUTPUT"
      continue
    fi
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_ENV/bin/python" \
      scripts/evaluate_oap.py \
      --config "$CONFIG" \
      --checkpoint "$CHECKPOINT" \
      --beta 0.1 \
      --output "$OUTPUT"
  done

  # Then calibrate inference beta on the final adapter checkpoint. beta=0.1
  # reuses the epoch-005 result above instead of encoding CIRR twice.
  for BETA in "${BETA_LIST[@]}"; do
    if [[ "$BETA" == "0.1" ]]; then
      continue
    fi
    CHECKPOINT="$RUN_DIR/checkpoint_epoch_005.pt"
    OUTPUT="$SWEEP_DIR/epoch_005_beta_${BETA}.json"
    if [[ -s "$OUTPUT" ]]; then
      echo "Already evaluated: $OUTPUT"
      continue
    fi
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_ENV/bin/python" \
      scripts/evaluate_oap.py \
      --config "$CONFIG" \
      --checkpoint "$CHECKPOINT" \
      --beta "$BETA" \
      --output "$OUTPUT"
  done
done

"$PYTHON_ENV/bin/python" scripts/summarize_oap_eval_sweep.py \
  --root outputs/oap_residual
