#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
benchmark_root="$(dirname "${project_root}")"
workspace_root="$(dirname "${benchmark_root}")"
config="${project_root}/configs/opd_clip_l_s2_centered_mse_e4.yaml"
output_root="${project_root}/outputs/opd_s2_centered_mse"
run_dir="${output_root}/clip_l_from_s2_centered_mse"
teacher_cache="${project_root}/outputs/on_policy_centered/cache/bge_mllm_s2_cirr_train"
fashioniq_root="${FASHIONIQ_ROOT:-${workspace_root}/CIR_new/datasets/fashion-iq}"
model_name="CLIP-L-OPD-S2-CenteredMSE"

cd "${project_root}"
mkdir -p "${output_root}"

echo "[stage 0/6] verify frozen S2 Teacher cache"
if [[ ! -f "${teacher_cache}/metadata.json" ]]; then
  echo "[error] missing S2 Teacher cache: ${teacher_cache}/metadata.json" >&2
  exit 1
fi

echo "[stage 1/6] train CLIP-L with random candidates and centered raw-similarity MSE"
final_checkpoint="${run_dir}/checkpoint_epoch_004.pt"
if [[ -f "${final_checkpoint}" ]]; then
  echo "[skip train] ${final_checkpoint}"
elif [[ -f "${run_dir}/last.pt" ]]; then
  echo "[resume train] ${run_dir}/last.pt"
  "${PYTHON}" scripts/train_on_policy_distillation.py \
    --config "${config}" \
    --resume-from "${run_dir}/last.pt"
else
  "${PYTHON}" scripts/train_on_policy_distillation.py --config "${config}"
fi

echo "[stage 2/6] evaluate every checkpoint on CIRR val"
for epoch in 001 002 003 004; do
  checkpoint="${run_dir}/checkpoint_epoch_${epoch}.pt"
  metrics="${run_dir}/val_epoch_${epoch}.json"
  if [[ ! -f "${metrics}" ]]; then
    "${PYTHON}" scripts/evaluate_on_policy_distillation.py \
      --config "${config}" \
      --checkpoint "${checkpoint}" \
      --output "${metrics}"
  fi
done

if [[ "${RUN_EXTENDED_EVAL:-1}" == "1" ]]; then
  echo "[stage 3/6] evaluate official PinPoint mAP@10 and NegRecall@10"
  (
    cd "${benchmark_root}/diag_model"
    "${PYTHON}" run_clip_pinpoint_official.py \
      --models "${model_name}" \
      --checkpoint "${model_name}=${final_checkpoint}" \
      --output-root "${output_root}/pinpoint_official" \
      --device cuda:0 \
      --batch-size "${PINPOINT_BATCH_SIZE:-128}"
  )

  echo "[stage 4/6] evaluate QDiag Delta, Stable Win, and random-10 separation"
  (
    cd "${benchmark_root}/diag_model"
    "${PYTHON}" evaluate_winrate_random10.py \
      --checkpoint "${model_name}=${final_checkpoint}" \
      --output-root "${output_root}/qdiag" \
      --device cuda:0 \
      --batch-size "${QDIAG_BATCH_SIZE:-128}" \
      --random-samples 10 \
      --seed 42
  )

  echo "[stage 5/6] evaluate FashionIQ transfer"
  fashion_output="${output_root}/fashioniq/clip_l_from_s2_centered_mse"
  if [[ ! -f "${fashion_output}/metrics.json" ]]; then
    "${PYTHON}" scripts/evaluate_fashioniq_clip.py \
      --model-name "${model_name}" \
      --config "${config}" \
      --checkpoint "${final_checkpoint}" \
      --dataset-root "${fashioniq_root}" \
      --output-dir "${fashion_output}" \
      --device cuda:0 \
      --image-batch-size "${FASHIONIQ_IMAGE_BATCH_SIZE:-256}" \
      --query-batch-size "${FASHIONIQ_QUERY_BATCH_SIZE:-256}" \
      --score-batch-size "${FASHIONIQ_SCORE_BATCH_SIZE:-1024}"
  fi
else
  echo "[skip stages 3-5] RUN_EXTENDED_EVAL=${RUN_EXTENDED_EVAL:-0}"
fi

echo "[stage 6/6] complete"
echo "[done] ${output_root}"
