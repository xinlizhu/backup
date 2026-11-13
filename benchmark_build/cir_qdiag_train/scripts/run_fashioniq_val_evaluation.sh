#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dataset_root="${FASHIONIQ_ROOT:-/mnt/data/datasets/search_dataset/zxl/CIR_new/datasets/fashion-iq}"
teacher_model_path="${TEACHER_MODEL_PATH:-/mnt/data/datasets/search_dataset/zxl/CIR_PLAY/models/checkpoints/BGE-VL-MLLM-S2}"
output_root="${OUTPUT_ROOT:-${project_root}/outputs/fashioniq_val}"
device="${DEVICE:-cuda:0}"

full_checkpoint="${project_root}/outputs/clip_l/full/baseline/checkpoint_epoch_010.pt"
rank_checkpoint="${project_root}/outputs/on_policy_live_top10/clip_l_original/positive_rank/checkpoint_epoch_008.pt"
centered_checkpoint="${project_root}/outputs/on_policy_live_top10/clip_l_original/centered_response/checkpoint_epoch_008.pt"
joint_checkpoint="${project_root}/outputs/on_policy_live_top10/clip_l_original/joint/checkpoint_epoch_008.pt"

required_paths=(
  "${dataset_root}/captions/cap.dress.val.json"
  "${dataset_root}/captions/cap.shirt.val.json"
  "${dataset_root}/captions/cap.toptee.val.json"
  "${dataset_root}/image_splits/split.dress.val.json"
  "${dataset_root}/image_splits/split.shirt.val.json"
  "${dataset_root}/image_splits/split.toptee.val.json"
  "${dataset_root}/images"
  "${project_root}/models/clip-vit-large-patch14"
  "${full_checkpoint}"
  "${rank_checkpoint}"
  "${centered_checkpoint}"
  "${joint_checkpoint}"
)
if [[ "${RUN_TEACHER:-1}" == "1" ]]; then
  required_paths+=("${teacher_model_path}")
fi
for required_path in "${required_paths[@]}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Missing required FashionIQ evaluation input: ${required_path}" >&2
    exit 1
  fi
done

mkdir -p "${output_root}"
cd "${project_root}"

run_clip() {
  local model_name="$1"
  local directory_name="$2"
  local config="$3"
  local checkpoint="${4:-}"
  local model_output="${output_root}/${directory_name}"
  if [[ -f "${model_output}/metrics.json" && "${FORCE_CACHE:-0}" != "1" ]]; then
    echo "[skip] ${model_name}: ${model_output}/metrics.json"
    return
  fi
  local checkpoint_args=()
  if [[ -n "${checkpoint}" ]]; then
    checkpoint_args=(--checkpoint "${checkpoint}")
  fi
  local force_args=()
  if [[ "${FORCE_CACHE:-0}" == "1" ]]; then
    force_args=(--force-cache)
  fi
  "${PYTHON}" scripts/evaluate_fashioniq_clip.py \
    --model-name "${model_name}" \
    --config "${config}" \
    "${checkpoint_args[@]}" \
    --dataset-root "${dataset_root}" \
    --output-dir "${model_output}" \
    --device "${device}" \
    --image-batch-size "${CLIP_IMAGE_BATCH_SIZE:-256}" \
    --query-batch-size "${CLIP_QUERY_BATCH_SIZE:-256}" \
    --score-batch-size "${SCORE_BATCH_SIZE:-1024}" \
    "${force_args[@]}"
}

echo "[1/6] CLIP-L original"
run_clip \
  "CLIP-L-original" \
  "clip_l_original" \
  "${project_root}/configs/on_policy_clip_l_rank.yaml"

echo "[2/6] CLIP-L Full InfoNCE baseline"
run_clip \
  "CLIP-L-Full-InfoNCE" \
  "clip_l_full_infonce" \
  "${project_root}/configs/clip_l_full_baseline.yaml" \
  "${full_checkpoint}"

echo "[3/6] CLIP-L On-Policy Positive Rank KD"
run_clip \
  "CLIP-L-OnPolicy-rank" \
  "clip_l_onpolicy_rank" \
  "${project_root}/configs/on_policy_clip_l_rank.yaml" \
  "${rank_checkpoint}"

echo "[4/6] CLIP-L On-Policy Centered Response KD"
run_clip \
  "CLIP-L-OnPolicy-centered" \
  "clip_l_onpolicy_centered" \
  "${project_root}/configs/on_policy_clip_l_centered_response.yaml" \
  "${centered_checkpoint}"

echo "[5/6] CLIP-L On-Policy Joint KD"
run_clip \
  "CLIP-L-OnPolicy-joint" \
  "clip_l_onpolicy_joint" \
  "${project_root}/configs/on_policy_clip_l_joint.yaml" \
  "${joint_checkpoint}"

if [[ "${RUN_TEACHER:-1}" == "1" ]]; then
  echo "[6/6] BGE-VL-MLLM-S2 Teacher"
  teacher_output="${output_root}/bge_vl_mllm_s2_teacher"
  if [[ -f "${teacher_output}/metrics.json" && "${FORCE_CACHE:-0}" != "1" ]]; then
    echo "[skip] BGE-VL-MLLM-S2 Teacher: ${teacher_output}/metrics.json"
  else
    teacher_force_args=()
    if [[ "${FORCE_CACHE:-0}" == "1" ]]; then
      teacher_force_args=(--force-cache)
    fi
    "${PYTHON}" scripts/evaluate_fashioniq_original_adapter.py \
      --model-name "BGE-VL-MLLM-S2 Teacher" \
      --adapter-name "BGE-VL-MLLM-S2" \
      --model-path "${teacher_model_path}" \
      --dataset-root "${dataset_root}" \
      --output-dir "${teacher_output}" \
      --device "${device}" \
      --image-batch-size "${TEACHER_IMAGE_BATCH_SIZE:-16}" \
      --query-batch-size "${TEACHER_QUERY_BATCH_SIZE:-16}" \
      --score-batch-size "${SCORE_BATCH_SIZE:-1024}" \
      "${teacher_force_args[@]}"
  fi
fi

summary_args=(
  --result "CLIP-L-original=${output_root}/clip_l_original/metrics.json"
  --result "CLIP-L-Full-InfoNCE=${output_root}/clip_l_full_infonce/metrics.json"
  --result "CLIP-L-OnPolicy-rank=${output_root}/clip_l_onpolicy_rank/metrics.json"
  --result "CLIP-L-OnPolicy-centered=${output_root}/clip_l_onpolicy_centered/metrics.json"
  --result "CLIP-L-OnPolicy-joint=${output_root}/clip_l_onpolicy_joint/metrics.json"
)
if [[ "${RUN_TEACHER:-1}" == "1" ]]; then
  summary_args+=(
    --result "BGE-VL-MLLM-S2 Teacher=${output_root}/bge_vl_mllm_s2_teacher/metrics.json"
  )
fi

"${PYTHON}" scripts/summarize_fashioniq.py \
  "${summary_args[@]}" \
  --output-dir "${output_root}"

echo "[done] ${output_root}/ALL_MODELS_RESULTS.md"
