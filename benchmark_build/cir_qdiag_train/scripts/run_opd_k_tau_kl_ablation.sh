#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
benchmark_root="$(dirname "${project_root}")"
workspace_root="$(dirname "${benchmark_root}")"
output_root="${project_root}/outputs/opd_k_tau_kl_ablation"
teacher_cache="${project_root}/outputs/on_policy_centered/cache/bge_mllm_s2_cirr_train"
teacher_model_path="${TEACHER_MODEL_PATH:-/mnt/data/datasets/search_dataset/zxl/CIR_PLAY/models/checkpoints/BGE-VL-MLLM-S2}"
fashioniq_root="${FASHIONIQ_ROOT:-${workspace_root}/CIR_new/datasets/fashion-iq}"

configs=(
  "${project_root}/configs/opd_clip_l_s2_k16_forward_t005.yaml"
  "${project_root}/configs/opd_clip_l_s2_k32_reverse_t005.yaml"
  "${project_root}/configs/opd_clip_l_s2_k32_forward_t1.yaml"
  "${project_root}/configs/opd_clip_l_s2_k32_reverse_t1.yaml"
  "${project_root}/configs/opd_clip_l_s2_k8_forward_t005.yaml"
)
run_names=(
  k16_forward_t005
  k32_reverse_t005
  k32_forward_t1
  k32_reverse_t1
  k8_forward_t005
)
model_names=(
  CLIP-L-OPD-S2-K16-FWD-T005
  CLIP-L-OPD-S2-K32-REV-T005
  CLIP-L-OPD-S2-K32-FWD-T1
  CLIP-L-OPD-S2-K32-REV-T1
  CLIP-L-OPD-S2-K8-FWD-T005
)

cd "${project_root}"
mkdir -p "${output_root}"
SECONDS=0

echo "[stage 0/7] verify/reuse frozen BGE-VL-MLLM-S2 Teacher cache"
if [[ -f "${teacher_cache}/metadata.json" ]]; then
  echo "[reuse] ${teacher_cache}"
else
  "${PYTHON}" scripts/prepare_on_policy_teacher_cache.py \
    --config "${configs[0]}" \
    --teacher-model-path "${teacher_model_path}" \
    --device cuda:0 \
    --image-batch-size "${TEACHER_IMAGE_BATCH_SIZE:-16}" \
    --query-batch-size "${TEACHER_QUERY_BATCH_SIZE:-16}"
fi

echo "[stage 1/7] train five new CLIP-L OPD controls"
for index in "${!run_names[@]}"; do
  run_dir="${output_root}/${run_names[$index]}"
  final_checkpoint="${run_dir}/checkpoint_epoch_004.pt"
  if [[ -f "${final_checkpoint}" ]]; then
    echo "[skip train] ${model_names[$index]}: ${final_checkpoint}"
  elif [[ -f "${run_dir}/last.pt" ]]; then
    echo "[resume train] ${model_names[$index]}: ${run_dir}/last.pt"
    "${PYTHON}" scripts/train_on_policy_distillation.py \
      --config "${configs[$index]}" \
      --resume-from "${run_dir}/last.pt"
  else
    echo "[train] ${model_names[$index]}"
    "${PYTHON}" scripts/train_on_policy_distillation.py \
      --config "${configs[$index]}"
  fi
done

echo "[stage 2/7] evaluate every saved checkpoint on CIRR val"
for index in "${!run_names[@]}"; do
  run_dir="${output_root}/${run_names[$index]}"
  for epoch in 001 002 003 004; do
    checkpoint="${run_dir}/checkpoint_epoch_${epoch}.pt"
    metrics="${run_dir}/val_epoch_${epoch}.json"
    if [[ -f "${metrics}" ]]; then
      echo "[skip CIRR] ${model_names[$index]} epoch ${epoch}"
    else
      "${PYTHON}" scripts/evaluate_on_policy_distillation.py \
        --config "${configs[$index]}" \
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
  if [[ -n "${model_csv}" ]]; then
    model_csv+=","
  fi
  model_csv+="${model_names[$index]}"
done

if [[ "${RUN_EXTENDED_EVAL:-1}" == "1" ]]; then
  echo "[stage 3/7] evaluate final checkpoints on official PinPoint"
  (
    cd "${benchmark_root}/diag_model"
    "${PYTHON}" run_clip_pinpoint_official.py \
      --models "${model_csv}" \
      "${checkpoint_args[@]}" \
      --output-root "${output_root}/pinpoint_official" \
      --device cuda:0 \
      --batch-size "${PINPOINT_BATCH_SIZE:-128}"
  )

  echo "[stage 4/7] evaluate final checkpoints on QDiag"
  (
    cd "${benchmark_root}/diag_model"
    "${PYTHON}" evaluate_winrate_random10.py \
      "${checkpoint_args[@]}" \
      --output-root "${output_root}/qdiag" \
      --device cuda:0 \
      --batch-size "${QDIAG_BATCH_SIZE:-128}" \
      --random-samples 10 \
      --seed 42
  )

  echo "[stage 5/7] evaluate final checkpoints on FashionIQ"
  for index in "${!run_names[@]}"; do
    fashion_output="${output_root}/fashioniq/${run_names[$index]}"
    metrics="${fashion_output}/metrics.json"
    if [[ -f "${metrics}" ]]; then
      echo "[skip FashionIQ] ${model_names[$index]}"
    else
      "${PYTHON}" scripts/evaluate_fashioniq_clip.py \
        --model-name "${model_names[$index]}" \
        --config "${configs[$index]}" \
        --checkpoint "${output_root}/${run_names[$index]}/checkpoint_epoch_004.pt" \
        --dataset-root "${fashioniq_root}" \
        --output-dir "${fashion_output}" \
        --device cuda:0 \
        --image-batch-size "${FASHIONIQ_IMAGE_BATCH_SIZE:-256}" \
        --query-batch-size "${FASHIONIQ_QUERY_BATCH_SIZE:-256}" \
        --score-batch-size "${FASHIONIQ_SCORE_BATCH_SIZE:-1024}"
    fi
  done
else
  echo "[skip stages 3-5] RUN_EXTENDED_EVAL=${RUN_EXTENDED_EVAL:-0}"
fi

echo "[stage 6/7] summarize K / KL / temperature ablation"
"${PYTHON}" scripts/summarize_opd_k_tau_kl_ablation.py \
  --output-root "${output_root}"

echo "[stage 7/7] complete"
echo "[runtime] total_seconds=${SECONDS}"
echo "[done] ${output_root}/RESULTS.md"
