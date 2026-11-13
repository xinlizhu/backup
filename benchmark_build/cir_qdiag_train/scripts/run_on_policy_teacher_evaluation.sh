#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
benchmark_root="$(dirname "${project_root}")"
teacher_model_path="${TEACHER_MODEL_PATH:-/mnt/data/datasets/search_dataset/zxl/CIR_PLAY/models/checkpoints/BGE-VL-MLLM-S2}"
output_root="${project_root}/outputs/on_policy_live_top10/teacher/BGE-VL-MLLM-S2"
device="${DEVICE:-cuda:0}"

if [[ ! -d "${teacher_model_path}" ]]; then
  echo "Missing Teacher model: ${teacher_model_path}" >&2
  exit 1
fi

mkdir -p "${output_root}"
cd "${project_root}"

echo "[stage 1/4] BGE-VL-MLLM-S2 Teacher: CIRR validation + Group Recall"
if [[ -f "${output_root}/cirr_val/val_metrics.json" ]]; then
  echo "[skip] ${output_root}/cirr_val/val_metrics.json"
else
  "${PYTHON}" scripts/evaluate_cirr_original_adapter.py \
    --model-name BGE-VL-MLLM-S2 \
    --model-path "${teacher_model_path}" \
    --output-dir "${output_root}/cirr_val" \
    --device "${device}" \
    --image-batch-size "${TEACHER_IMAGE_BATCH_SIZE:-16}" \
    --query-batch-size "${TEACHER_QUERY_BATCH_SIZE:-16}" \
    --score-batch-size "${SCORE_BATCH_SIZE:-512}"
fi

echo "[stage 2/4] BGE-VL-MLLM-S2 Teacher: official PinPoint mAP@10 + NegRecall@10"
if [[ -f "${output_root}/pinpoint_official/official_map10_negrecall10.csv" ]]; then
  echo "[skip] ${output_root}/pinpoint_official/official_map10_negrecall10.csv"
else
  (
    cd "${benchmark_root}/diag_model"
    "${PYTHON}" run_clip_pinpoint_official.py \
      --models BGE-VL-MLLM-S2 \
      --output-root "${output_root}/pinpoint_official" \
      --device "${device}" \
      --batch-size "${PINPOINT_BATCH_SIZE:-16}" \
      --num-workers "${PINPOINT_NUM_WORKERS:-8}"
  )
fi

echo "[stage 3/4] BGE-VL-MLLM-S2 Teacher: Delta/Win/Stable-Win/random-10"
if [[ -f "${output_root}/pinpoint_diagnostics/combined_summary.csv" ]]; then
  echo "[skip] ${output_root}/pinpoint_diagnostics/combined_summary.csv"
else
  (
    cd "${benchmark_root}/diag_model"
    "${PYTHON}" evaluate_winrate_random10.py \
      --pretrained-model "BGE-VL-MLLM-S2=${teacher_model_path}" \
      --output-root "${output_root}/pinpoint_diagnostics" \
      --device "${device}" \
      --batch-size "${PINPOINT_BATCH_SIZE:-16}" \
      --num-workers "${PINPOINT_NUM_WORKERS:-8}" \
      --random-samples 10
  )
fi

echo "[stage 4/4] merge Teacher and existing CLIP-L results"
"${PYTHON}" scripts/summarize_on_policy_with_teacher.py

echo "[done] ${project_root}/outputs/on_policy_live_top10/ALL_MODELS_RESULTS.md"
