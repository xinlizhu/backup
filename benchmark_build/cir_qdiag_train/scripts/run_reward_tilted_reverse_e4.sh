#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
benchmark_root="$(dirname "${project_root}")"
workspace_root="$(dirname "${benchmark_root}")"
output_root="${project_root}/outputs/reward_tilted_reverse_beta0_1_e4"
fashioniq_root="${FASHIONIQ_ROOT:-${workspace_root}/CIR_new/datasets/fashion-iq}"
run_indices_text="${RUN_INDICES:-0 1}"
run_summary="${RUN_SUMMARY:-1}"

configs=(
  "${project_root}/configs/clip_l_reward_tilted_reverse_opd_e4.yaml"
  "${project_root}/configs/clip_l_reward_tilted_reverse_gspo_e4.yaml"
)
run_names=(reverse_opd reverse_gspo)
model_names=(
  CLIP-L-Reward-Tilted-Reverse-OPD
  CLIP-L-Reward-Tilted-Reverse-KL-GSPO
)

read -r -a run_indices <<< "${run_indices_text}"
if [[ "${#run_indices[@]}" -eq 0 ]]; then
  echo "[error] RUN_INDICES must contain 0 and/or 1" >&2
  exit 1
fi
declare -A seen_indices=()
for index in "${run_indices[@]}"; do
  if [[ ! "${index}" =~ ^[01]$ ]]; then
    echo "[error] invalid RUN_INDICES entry '${index}'; expected 0 or 1" >&2
    exit 1
  fi
  if [[ -n "${seen_indices[$index]:-}" ]]; then
    echo "[error] duplicate RUN_INDICES entry '${index}'" >&2
    exit 1
  fi
  seen_indices[$index]=1
done

cd "${project_root}"
mkdir -p "${output_root}"
SECONDS=0

echo "[stage 0/5] verify selected reverse-KL configurations"
for index in "${run_indices[@]}"; do
  "${PYTHON}" -c \
    'import sys; from cir_qdiag_train.on_policy_config import load_on_policy_config; c=load_on_policy_config(sys.argv[1]); assert c.experiment.initial_checkpoint is None; assert c.objective.rank_kl_direction == "reverse"; assert c.objective.reward_tilt_beta == 0.1; assert not c.objective.anchor_kl_enabled; print(f"[config] {c.experiment.name}: {c.objective.loss_type}, original={c.model.pretrained_model_name_or_path}, updates={c.objective.policy_update_epochs}")' \
    "${configs[$index]}"
done

echo "[stage 1/5] train selected original-CLIP-L models"
for index in "${run_indices[@]}"; do
  run_dir="${output_root}/${run_names[$index]}"
  final_checkpoint="${run_dir}/checkpoint_epoch_004.pt"
  if [[ -f "${final_checkpoint}" ]]; then
    echo "[skip train] ${model_names[$index]}"
  elif [[ -f "${run_dir}/last.pt" ]]; then
    echo "[resume train] ${model_names[$index]}"
    "${PYTHON}" scripts/train_on_policy_distillation.py \
      --config "${configs[$index]}" \
      --resume-from "${run_dir}/last.pt" \
      --batch-size-per-gpu "${TRAIN_BATCH_SIZE:-32}" \
      --num-workers "${NUM_WORKERS:-8}" \
      --mining-image-batch-size "${MINING_IMAGE_BATCH_SIZE:-1024}"
  else
    echo "[train] ${model_names[$index]}"
    "${PYTHON}" scripts/train_on_policy_distillation.py \
      --config "${configs[$index]}" \
      --batch-size-per-gpu "${TRAIN_BATCH_SIZE:-32}" \
      --num-workers "${NUM_WORKERS:-8}" \
      --mining-image-batch-size "${MINING_IMAGE_BATCH_SIZE:-1024}"
  fi
done

echo "[stage 2/5] evaluate every checkpoint on CIRR val"
for index in "${run_indices[@]}"; do
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

echo "[stage 3/5] evaluate every checkpoint on FashionIQ val"
for index in "${run_indices[@]}"; do
  for epoch in 001 002 003 004; do
    fashion_output="${output_root}/fashioniq/${run_names[$index]}/epoch_${epoch}"
    metrics="${fashion_output}/metrics.json"
    if [[ -f "${metrics}" ]]; then
      echo "[skip FashionIQ] ${model_names[$index]} epoch ${epoch}"
    else
      "${PYTHON}" scripts/evaluate_fashioniq_clip.py \
        --model-name "${model_names[$index]}-E${epoch}" \
        --config "${configs[$index]}" \
        --checkpoint "${output_root}/${run_names[$index]}/checkpoint_epoch_${epoch}.pt" \
        --dataset-root "${fashioniq_root}" \
        --output-dir "${fashion_output}" \
        --device cuda:0 \
        --image-batch-size "${FASHIONIQ_IMAGE_BATCH_SIZE:-256}" \
        --query-batch-size "${FASHIONIQ_QUERY_BATCH_SIZE:-256}" \
        --score-batch-size "${FASHIONIQ_SCORE_BATCH_SIZE:-1024}"
    fi
  done
done

echo "[stage 4/5] summarize completed evaluations"
if [[ "${run_summary}" == "1" ]]; then
  "${PYTHON}" scripts/summarize_reward_tilted_reverse.py \
    --output-root "${output_root}"
else
  echo "[skip summary] RUN_SUMMARY=${run_summary}"
fi

echo "[stage 5/5] complete"
echo "[runtime] total_seconds=${SECONDS}"
echo "[done] ${output_root}/RESULTS.md"
