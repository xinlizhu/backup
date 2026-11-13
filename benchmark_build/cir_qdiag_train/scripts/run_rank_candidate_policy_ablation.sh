#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
benchmark_root="$(dirname "${project_root}")"
output_root="${project_root}/outputs/rank_candidate_policy_e4"
teacher_model_path="${TEACHER_MODEL_PATH:-/mnt/data/datasets/search_dataset/zxl/CIR_PLAY/models/checkpoints/BGE-VL-MLLM-S2}"
fashioniq_root="${FASHIONIQ_ROOT:-/mnt/data/datasets/search_dataset/zxl/CIR_new/datasets/fashion-iq}"

run_names=(
  dynamic_student_rank
  teacher_topk_rank
  random_rank
  static_student_rank
  dynamic_student_infonce
  dynamic_student_infonce_teacher_top10
)
model_names=(
  DynamicStudentRank
  TeacherTopKRank
  RandomRank
  StaticStudentRank
  DynamicStudentInfoNCE
  DynamicStudentInfoNCETeacherTop10
)
configs=(
  "${project_root}/configs/rank_ablation_dynamic_student_rank_e4.yaml"
  "${project_root}/configs/rank_ablation_teacher_topk_rank_e4.yaml"
  "${project_root}/configs/rank_ablation_random_rank_e4.yaml"
  "${project_root}/configs/rank_ablation_static_student_rank_e4.yaml"
  "${project_root}/configs/rank_ablation_dynamic_student_infonce_e4.yaml"
  "${project_root}/configs/rank_ablation_dynamic_student_infonce_teacher_top10_e4.yaml"
)

mkdir -p "${output_root}"
cd "${project_root}"

echo "[stage 1/6] verify/cache frozen BGE Teacher features"
"${PYTHON}" scripts/prepare_on_policy_teacher_cache.py \
  --config "${configs[0]}" \
  --teacher-model-path "${teacher_model_path}" \
  --device cuda:0 \
  --image-batch-size "${TEACHER_IMAGE_BATCH_SIZE:-16}" \
  --query-batch-size "${TEACHER_QUERY_BATCH_SIZE:-16}"

echo "[stage 2/6] train controlled Rank candidate-policy ablation"
for index in "${!run_names[@]}"; do
  run_name="${run_names[$index]}"
  config="${configs[$index]}"
  run_dir="${output_root}/${run_name}"
  final_checkpoint="${run_dir}/checkpoint_epoch_004.pt"
  if [[ -f "${final_checkpoint}" ]]; then
    echo "[skip train] ${run_name}: ${final_checkpoint}"
  elif [[ -f "${run_dir}/last.pt" ]]; then
    echo "[resume train] ${run_name}: ${run_dir}/last.pt"
    "${PYTHON}" scripts/train_on_policy_distillation.py \
      --config "${config}" \
      --resume-from "${run_dir}/last.pt"
  else
    echo "[train] ${run_name}"
    "${PYTHON}" scripts/train_on_policy_distillation.py --config "${config}"
  fi
done

echo "[stage 3/6] evaluate every saved epoch on CIRR val"
for index in "${!run_names[@]}"; do
  run_name="${run_names[$index]}"
  config="${configs[$index]}"
  for epoch in 001 002 003 004; do
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

checkpoint_args=()
model_csv=""
for index in "${!run_names[@]}"; do
  checkpoint="${output_root}/${run_names[$index]}/checkpoint_epoch_004.pt"
  checkpoint_args+=(--checkpoint "${model_names[$index]}=${checkpoint}")
  if [[ -z "${model_csv}" ]]; then
    model_csv="${model_names[$index]}"
  else
    model_csv+=",${model_names[$index]}"
  fi
done

if [[ "${RUN_EXTENDED_EVAL:-1}" == "1" ]]; then
  echo "[stage 4/6] evaluate final checkpoints on official PinPoint"
  (
    cd "${benchmark_root}/diag_model"
    "${PYTHON}" run_clip_pinpoint_official.py \
      --models "${model_csv}" \
      "${checkpoint_args[@]}" \
      --output-root "${output_root}/pinpoint_official" \
      --device cuda:0 \
      --batch-size "${PINPOINT_BATCH_SIZE:-128}"
  )

  echo "[stage 5/6] evaluate final checkpoints on QDiag"
  (
    cd "${benchmark_root}/diag_model"
    "${PYTHON}" evaluate_winrate_random10.py \
      "${checkpoint_args[@]}" \
      --output-root "${output_root}/pinpoint_diagnostics" \
      --device cuda:0 \
      --batch-size "${PINPOINT_BATCH_SIZE:-128}" \
      --random-samples 10
  )

  echo "[stage 6/6] evaluate final checkpoints on FashionIQ"
  fashion_output="${output_root}/fashioniq"
  mkdir -p "${fashion_output}"
  fashion_summary_args=()
  for index in "${!run_names[@]}"; do
    model_name="${model_names[$index]}"
    model_output="${fashion_output}/${run_names[$index]}"
    metrics="${model_output}/metrics.json"
    if [[ ! -f "${metrics}" ]]; then
      "${PYTHON}" scripts/evaluate_fashioniq_clip.py \
        --model-name "${model_name}" \
        --config "${configs[$index]}" \
        --checkpoint "${output_root}/${run_names[$index]}/checkpoint_epoch_004.pt" \
        --dataset-root "${fashioniq_root}" \
        --output-dir "${model_output}" \
        --device cuda:0 \
        --image-batch-size "${FASHIONIQ_IMAGE_BATCH_SIZE:-256}" \
        --query-batch-size "${FASHIONIQ_QUERY_BATCH_SIZE:-256}" \
        --score-batch-size "${FASHIONIQ_SCORE_BATCH_SIZE:-1024}"
    fi
    fashion_summary_args+=(--result "${model_name}=${metrics}")
  done
  "${PYTHON}" scripts/summarize_fashioniq.py \
    "${fashion_summary_args[@]}" \
    --output-dir "${fashion_output}"
else
  echo "[skip stages 4-6] RUN_EXTENDED_EVAL=${RUN_EXTENDED_EVAL:-0}"
fi

"${PYTHON}" scripts/summarize_rank_candidate_ablation.py \
  --output-root "${output_root}"

echo "[done] ${output_root}/RESULTS.md"
