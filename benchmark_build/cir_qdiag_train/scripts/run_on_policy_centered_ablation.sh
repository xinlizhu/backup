#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
benchmark_root="$(dirname "${project_root}")"
config_dir="${project_root}/configs"
output_root="${project_root}/outputs/on_policy_live_top10/clip_l_original"
teacher_model_path="${TEACHER_MODEL_PATH:-/mnt/data/datasets/search_dataset/zxl/CIR_PLAY/models/checkpoints/BGE-VL-MLLM-S2}"
clip_model_path="${project_root}/models/clip-vit-large-patch14"

rank_config="${config_dir}/on_policy_clip_l_rank.yaml"
run_names=(positive_rank centered_response joint)
configs=(
  "${rank_config}"
  "${config_dir}/on_policy_clip_l_centered_response.yaml"
  "${config_dir}/on_policy_clip_l_joint.yaml"
)

if [[ ! -d "${teacher_model_path}" ]]; then
  echo "Missing Teacher model: ${teacher_model_path}" >&2
  exit 1
fi
if [[ ! -d "${clip_model_path}" ]]; then
  echo "Missing original CLIP-L model: ${clip_model_path}" >&2
  exit 1
fi
cd "${project_root}"

echo "[stage 1/5] cache frozen BGE Teacher for all original CIRR train positives"
"${PYTHON}" scripts/prepare_on_policy_teacher_cache.py \
  --config "${rank_config}" \
  --teacher-model-path "${teacher_model_path}" \
  --device cuda:0 \
  --image-batch-size "${TEACHER_IMAGE_BATCH_SIZE:-16}" \
  --query-batch-size "${TEACHER_QUERY_BATCH_SIZE:-16}"

echo "[stage 2/5] train three live-candidate, Teacher-Top10 8-epoch variants from original CLIP-L"
for index in "${!run_names[@]}"; do
  run_name="${run_names[$index]}"
  config="${configs[$index]}"
  run_dir="${output_root}/${run_name}"
  final_checkpoint="${run_dir}/checkpoint_epoch_008.pt"
  if [[ -f "${final_checkpoint}" ]]; then
    echo "[skip train] ${run_name}: ${final_checkpoint}"
  elif [[ -f "${run_dir}/last.pt" ]]; then
    echo "[resume train] ${run_name}: ${run_dir}/last.pt"
    "${PYTHON}" scripts/train_on_policy_distillation.py \
      --config "${config}" \
      --resume-from "${run_dir}/last.pt"
  else
    echo "[train from original CLIP-L] ${run_name}"
    "${PYTHON}" scripts/train_on_policy_distillation.py --config "${config}"
  fi
done

echo "[stage 3/5] evaluate CIRR validation and Group Recall"
baseline_metrics="${output_root}/baseline/val_metrics.json"
mkdir -p "$(dirname "${baseline_metrics}")"
if [[ ! -f "${baseline_metrics}" ]]; then
  "${PYTHON}" scripts/evaluate_on_policy_distillation.py \
    --config "${rank_config}" \
    --output "${baseline_metrics}"
fi
if [[ "${EVAL_CURVE:-1}" == "1" ]]; then
  eval_epochs=(004 008)
else
  eval_epochs=(008)
fi
for index in "${!run_names[@]}"; do
  run_name="${run_names[$index]}"
  config="${configs[$index]}"
  for epoch in "${eval_epochs[@]}"; do
    checkpoint="${output_root}/${run_name}/checkpoint_epoch_${epoch}.pt"
    metrics="${output_root}/${run_name}/val_epoch_${epoch}.json"
    if [[ ! -f "${metrics}" ]]; then
      "${PYTHON}" scripts/evaluate_on_policy_distillation.py \
        --config "${config}" \
        --checkpoint "${checkpoint}" \
        --output "${metrics}"
    fi
  done
done

model_names="CLIP-L-original,CLIP-L-OnPolicy-rank,CLIP-L-OnPolicy-centered,CLIP-L-OnPolicy-joint"
checkpoint_args=(
  --checkpoint "CLIP-L-OnPolicy-rank=${output_root}/positive_rank/checkpoint_epoch_008.pt"
  --checkpoint "CLIP-L-OnPolicy-centered=${output_root}/centered_response/checkpoint_epoch_008.pt"
  --checkpoint "CLIP-L-OnPolicy-joint=${output_root}/joint/checkpoint_epoch_008.pt"
)

echo "[stage 4/5] evaluate official PinPoint mAP@10 and NegRecall@10"
if [[ "${RUN_PINPOINT_OFFICIAL:-1}" == "1" ]]; then
  (
    cd "${benchmark_root}/diag_model"
    "${PYTHON}" run_clip_pinpoint_official.py \
      --models "${model_names}" \
      "${checkpoint_args[@]}" \
      --output-root "${output_root}/pinpoint_official" \
      --device cuda:0 \
      --batch-size "${PINPOINT_BATCH_SIZE:-128}"
  )
fi

echo "[stage 5/5] evaluate PinPoint Delta, Stable Win, and random-10 separation"
if [[ "${RUN_PINPOINT_DIAGNOSTICS:-1}" == "1" ]]; then
  (
    cd "${benchmark_root}/diag_model"
    "${PYTHON}" evaluate_winrate_random10.py \
      --pretrained-clip "CLIP-L-original=${clip_model_path}" \
      "${checkpoint_args[@]}" \
      --output-root "${output_root}/pinpoint_diagnostics" \
      --device cuda:0 \
      --batch-size "${PINPOINT_BATCH_SIZE:-128}" \
      --random-samples 10
  )
fi

"${PYTHON}" scripts/summarize_on_policy_centered_ablation.py \
  --output-root "${output_root}"

echo "[done] ${output_root}/ABLATION_SUMMARY.md"
