#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
benchmark_root="$(dirname "${project_root}")"
experiment_root="${project_root}/outputs/rank_candidate_policy_e4"
diagnostic_root="${experiment_root}/pinpoint_diagnostics"

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

checkpoint_args=()
for index in "${!run_names[@]}"; do
  run_name="${run_names[$index]}"
  base_name="${model_names[$index]}"
  for epoch in 001 002 003 004; do
    checkpoint="${experiment_root}/${run_name}/checkpoint_epoch_${epoch}.pt"
    if [[ ! -f "${checkpoint}" ]]; then
      echo "[error] missing checkpoint: ${checkpoint}" >&2
      exit 1
    fi
    if [[ "${epoch}" == "004" ]]; then
      # Reuse the six epoch-4 caches produced by the original final evaluation.
      eval_name="${base_name}"
    else
      eval_name="${base_name}_E${epoch:1:2}"
    fi
    checkpoint_args+=(--checkpoint "${eval_name}=${checkpoint}")
  done
done

mkdir -p "${diagnostic_root}"

echo "[stage 1/2] QDiag: six ablations x four epochs (24 checkpoints)"
(
  cd "${benchmark_root}/diag_model"
  "${PYTHON}" evaluate_winrate_random10.py \
    "${checkpoint_args[@]}" \
    --output-root "${diagnostic_root}" \
    --device cuda:0 \
    --batch-size "${PINPOINT_BATCH_SIZE:-128}" \
    --num-workers "${PINPOINT_NUM_WORKERS:-32}" \
    --random-samples 10 \
    --bootstrap-replicates "${QDIAG_BOOTSTRAP_REPLICATES:-2000}" \
    --seed 42
)

echo "[stage 2/2] summarize per-epoch QDiag results"
cd "${project_root}"
"${PYTHON}" scripts/summarize_qdiag_ablation_epochs.py \
  --input "${diagnostic_root}/combined_summary.csv" \
  --output "${experiment_root}/QDIAG_ALL_EPOCHS.md" \
  --csv-output "${experiment_root}/qdiag_all_epochs_summary.csv"

echo "[done] ${experiment_root}/QDIAG_ALL_EPOCHS.md"
