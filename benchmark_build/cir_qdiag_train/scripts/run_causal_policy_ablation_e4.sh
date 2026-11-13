#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
benchmark_root="$(dirname "${project_root}")"
workspace_root="$(dirname "${benchmark_root}")"
output_root="${project_root}/outputs/causal_policy_ablation_e4"
initial_checkpoint="${project_root}/outputs/rank_candidate_policy_e4/dynamic_student_rank/checkpoint_epoch_004.pt"
fashioniq_root="${FASHIONIQ_ROOT:-${workspace_root}/CIR_new/datasets/fashion-iq}"

configs=(
  "${project_root}/configs/opd_clip_l_cp_reinforce_e4.yaml"
  "${project_root}/configs/opd_clip_l_cp_grpo_e4.yaml"
  "${project_root}/configs/opd_clip_l_cp_gspo_noclip_e4.yaml"
  "${project_root}/configs/opd_clip_l_cp_gspo_e4.yaml"
)
run_names=(
  cp_reinforce
  cp_grpo
  cp_gspo_noclip
  cp_gspo
)
model_names=(
  CLIP-L-OPD-CP-Reinforce
  CLIP-L-OPD-CP-GRPO
  CLIP-L-OPD-CP-GSPO-NoClip
  CLIP-L-OPD-CP-GSPO
)

cd "${project_root}"
mkdir -p "${output_root}"
SECONDS=0

echo "[stage 0/7] verify common initialization and four causal objectives"
if [[ ! -f "${initial_checkpoint}" ]]; then
  echo "[error] missing OPD E4 checkpoint: ${initial_checkpoint}" >&2
  exit 1
fi
for config in "${configs[@]}"; do
  "${PYTHON}" -c \
    'import sys; from cir_qdiag_train.on_policy_config import load_on_policy_config; c=load_on_policy_config(sys.argv[1]); assert c.objective.is_causal_policy_loss; assert not c.model.freeze_backbone; assert not c.model.freeze_vision_encoder; print(f"[config] {c.experiment.name}: {c.objective.loss_type}, updates={c.objective.policy_update_epochs}, clip={c.objective.policy_clip_epsilon}")' \
    "${config}"
done

echo "[stage 1/7] train four OPD-initialized causal-prefix policies"
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

reference_candidates="${output_root}/cp_reinforce/on_policy/epoch_001/candidate_indices.npy"
for run_name in "${run_names[@]:1}"; do
  candidate_file="${output_root}/${run_name}/on_policy/epoch_001/candidate_indices.npy"
  if ! cmp -s "${reference_candidates}" "${candidate_file}"; then
    echo "[error] epoch-1 candidates differ: cp_reinforce vs ${run_name}" >&2
    exit 1
  fi
done
echo "[matched control] all epoch-1 Top-32 candidate pools are byte-identical"

echo "[stage 2/7] evaluate every checkpoint on CIRR val"
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

  echo "[stage 4/7] evaluate final checkpoints on CIR-QDiag"
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

echo "[stage 6/7] summarize causal-prefix policy ablation"
"${PYTHON}" scripts/summarize_causal_policy_ablation.py \
  --output-root "${output_root}"

echo "[stage 7/7] complete"
echo "[runtime] total_seconds=${SECONDS}"
echo "[done] ${output_root}/RESULTS.md"
