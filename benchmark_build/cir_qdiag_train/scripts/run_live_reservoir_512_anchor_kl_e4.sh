#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
benchmark_root="$(dirname "${project_root}")"
workspace_root="$(dirname "${benchmark_root}")"
output_root="${project_root}/outputs/live_reservoir_512_anchor_kl_e4_cached"
initial_checkpoint="${project_root}/outputs/rank_candidate_policy_e4/dynamic_student_rank/checkpoint_epoch_004.pt"
fashioniq_root="${FASHIONIQ_ROOT:-${workspace_root}/CIR_new/datasets/fashion-iq}"
screen_batch_size="${LIVE_SELECTION_IMAGE_BATCH_SIZE:-1024}"
mining_image_batch_size="${MINING_IMAGE_BATCH_SIZE:-1024}"
train_batch_size="${TRAIN_BATCH_SIZE:-32}"
num_workers="${NUM_WORKERS:-8}"
pixel_cache_build_batch_size="${PIXEL_CACHE_BUILD_BATCH_SIZE:-128}"
pixel_cache_workers="${PIXEL_CACHE_WORKERS:-12}"
pixel_cache_build_prefetch_factor="${PIXEL_CACHE_BUILD_PREFETCH_FACTOR:-3}"
pixel_cache_transfer_prefetch_batches="${PIXEL_CACHE_TRANSFER_PREFETCH_BATCHES:-2}"
pixel_cache_dtype="${PIXEL_CACHE_DTYPE:-float16}"
run_indices_text="${RUN_INDICES:-0 1 2 3}"
run_summary="${RUN_SUMMARY:-1}"

configs=(
  "${project_root}/configs/opd_clip_l_live_r512_grpo_e4.yaml"
  "${project_root}/configs/opd_clip_l_live_r512_grpo_anchor_kl_e4.yaml"
  "${project_root}/configs/opd_clip_l_live_r512_gspo_e4.yaml"
  "${project_root}/configs/opd_clip_l_live_r512_gspo_anchor_kl_e4.yaml"
)
run_names=(
  grpo_no_anchor_kl
  grpo_adaptive_anchor_kl
  gspo_no_anchor_kl
  gspo_adaptive_anchor_kl
)
model_names=(
  CLIP-L-R512-GRPO
  CLIP-L-R512-GRPO-AnchorKL
  CLIP-L-R512-GSPO
  CLIP-L-R512-GSPO-AnchorKL
)

read -r -a run_indices <<< "${run_indices_text}"
if [[ "${#run_indices[@]}" -eq 0 ]]; then
  echo "[error] RUN_INDICES must contain at least one index from: 0 1 2 3" >&2
  exit 1
fi
declare -A seen_indices=()
for index in "${run_indices[@]}"; do
  if [[ ! "${index}" =~ ^[0-3]$ ]]; then
    echo "[error] invalid RUN_INDICES entry '${index}'; expected: 0 1 2 3" >&2
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

echo "[selection] RUN_INDICES=${run_indices[*]}"
for index in "${run_indices[@]}"; do
  echo "[selection] ${index}: ${model_names[$index]}"
done

echo "[stage 0/5] verify selected teacher-free live-reservoir configurations"
if [[ ! -f "${initial_checkpoint}" ]]; then
  echo "[error] missing common initialization: ${initial_checkpoint}" >&2
  exit 1
fi
for index in "${run_indices[@]}"; do
  config="${configs[$index]}"
  "${PYTHON}" -c \
    'import sys; from cir_qdiag_train.on_policy_config import load_on_policy_config; c=load_on_policy_config(sys.argv[1]); assert c.mining.live_reservoir_selection and c.mining.pixel_cache_enabled and c.mining.reservoir_top_k == 512 and c.mining.top_k == 32 and not c.mining.persist_candidate_cache and c.objective.policy_update_epochs == 2 and not c.objective.needs_teacher_scores; print(f"[config] {c.experiment.name}: {c.objective.loss_type}, anchor_kl={c.objective.anchor_kl_enabled}, pixel_cache={c.mining.pixel_cache_dtype}")' \
    "${config}"
done

echo "[stage 1/5] train selected Top-512 -> live Top-32 policies"
for index in "${run_indices[@]}"; do
  run_dir="${output_root}/${run_names[$index]}"
  final_checkpoint="${run_dir}/checkpoint_epoch_004.pt"
  train_args=(
    --config "${configs[$index]}"
    --live-selection-image-batch-size "${screen_batch_size}"
    --mining-image-batch-size "${mining_image_batch_size}"
    --pixel-cache-build-batch-size "${pixel_cache_build_batch_size}"
    --pixel-cache-workers "${pixel_cache_workers}"
    --pixel-cache-build-prefetch-factor "${pixel_cache_build_prefetch_factor}"
    --pixel-cache-transfer-prefetch-batches "${pixel_cache_transfer_prefetch_batches}"
    --pixel-cache-dtype "${pixel_cache_dtype}"
    --batch-size-per-gpu "${train_batch_size}"
    --num-workers "${num_workers}"
  )
  if [[ -f "${final_checkpoint}" ]]; then
    echo "[skip train] ${model_names[$index]}"
  elif [[ -f "${run_dir}/last.pt" ]]; then
    echo "[resume train] ${model_names[$index]}"
    "${PYTHON}" scripts/train_on_policy_distillation.py \
      "${train_args[@]}" \
      --resume-from "${run_dir}/last.pt"
  else
    echo "[train] ${model_names[$index]}"
    "${PYTHON}" scripts/train_on_policy_distillation.py "${train_args[@]}"
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

echo "[stage 4/5] summarize CIRR and FashionIQ curves"
if [[ "${run_summary}" == "1" ]]; then
  "${PYTHON}" scripts/summarize_live_reservoir_512.py \
    --output-root "${output_root}"
else
  echo "[skip summary] RUN_SUMMARY=${run_summary}"
fi

echo "[stage 5/5] complete"
echo "[runtime] total_seconds=${SECONDS}"
echo "[done] ${output_root}/RESULTS.md"
