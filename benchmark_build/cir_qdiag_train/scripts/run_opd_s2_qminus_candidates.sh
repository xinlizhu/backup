#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
benchmark_root="$(dirname "${project_root}")"
workspace_root="$(dirname "${benchmark_root}")"
output_root="${project_root}/outputs/opd_s2_qminus_candidates"
teacher_cache="${project_root}/outputs/on_policy_centered/cache/bge_mllm_s2_cirr_train"
negative_source="${benchmark_root}/cir_qdiag/outputs/train_all_v5/final_no_va.jsonl"
negative_manifest="${project_root}/data/cirr_train_all_v5_no_va.jsonl"
fashioniq_root="${FASHIONIQ_ROOT:-${workspace_root}/CIR_new/datasets/fashion-iq}"

configs=(
  "${project_root}/configs/opd_clip_l_s2_qminus_candidates_rank_k16.yaml"
  "${project_root}/configs/opd_clip_l_s2_qminus_candidates_rank.yaml"
  "${project_root}/configs/opd_clip_l_s2_qminus_candidates_rank_k64.yaml"
)
run_names=(
  clip_l_from_s2_mixed_qminus_top16
  clip_l_from_s2_mixed_qminus_top32
  clip_l_from_s2_mixed_qminus_top64
)
model_names=(
  CLIP-L-OPD-S2-QminusCandidates-K16
  CLIP-L-OPD-S2-QminusCandidates-K32
  CLIP-L-OPD-S2-QminusCandidates-K64
)

cd "${project_root}"
mkdir -p "${output_root}"

echo "[stage 0/7] verify CIR-QDiag train q- data and frozen S2 positive cache"
for required in \
  "${negative_source}" \
  "${negative_manifest}" \
  "${teacher_cache}/metadata.json"; do
  if [[ ! -f "${required}" ]]; then
    echo "[error] missing required artifact: ${required}" >&2
    exit 1
  fi
done

echo "[stage 1/7] cache frozen S2 composed embeddings for all CIR-QDiag train q-"
"${PYTHON}" scripts/prepare_on_policy_teacher_cache.py \
  --config "${configs[1]}" \
  --device cuda:0 \
  --image-batch-size "${S2_IMAGE_BATCH_SIZE:-16}" \
  --query-batch-size "${S2_QMINUS_BATCH_SIZE:-16}"

echo "[stage 2/7] train K=16, 32, 64 CLIP-L experiments sequentially"
for index in "${!run_names[@]}"; do
  run_dir="${output_root}/${run_names[$index]}"
  checkpoint="${run_dir}/checkpoint_epoch_004.pt"
  if [[ -f "${checkpoint}" ]]; then
    echo "[skip train] ${model_names[$index]}: ${checkpoint}"
  elif [[ -f "${run_dir}/last.pt" ]]; then
    echo "[resume train] ${model_names[$index]}"
    "${PYTHON}" scripts/train_on_policy_distillation.py \
      --config "${configs[$index]}" \
      --resume-from "${run_dir}/last.pt"
  else
    echo "[train] ${model_names[$index]}"
    "${PYTHON}" scripts/train_on_policy_distillation.py \
      --config "${configs[$index]}"
  fi
done

echo "[stage 3/7] evaluate all four CIRR checkpoints for K=16, 32, 64"
for index in "${!run_names[@]}"; do
  run_dir="${output_root}/${run_names[$index]}"
  for epoch in 001 002 003 004; do
    epoch_checkpoint="${run_dir}/checkpoint_epoch_${epoch}.pt"
    epoch_metrics="${run_dir}/val_epoch_${epoch}.json"
    if [[ ! -f "${epoch_metrics}" ]]; then
      "${PYTHON}" scripts/evaluate_on_policy_distillation.py \
        --config "${configs[$index]}" \
        --checkpoint "${epoch_checkpoint}" \
        --output "${epoch_metrics}"
    fi
  done
done

if [[ "${RUN_EXTENDED_EVAL:-1}" == "1" ]]; then
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

  echo "[stage 4/7] evaluate official PinPoint mAP@10 and NegRecall@10"
  (
    cd "${benchmark_root}/diag_model"
    "${PYTHON}" run_clip_pinpoint_official.py \
      --models "${model_csv}" \
      "${checkpoint_args[@]}" \
      --output-root "${output_root}/pinpoint_official" \
      --device cuda:0 \
      --batch-size "${PINPOINT_BATCH_SIZE:-128}"
  )

  echo "[stage 5/7] evaluate QDiag Delta, Stable Win, and random-10 separation"
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

  echo "[stage 6/7] evaluate FashionIQ transfer"
  for index in "${!run_names[@]}"; do
    fashion_output="${output_root}/fashioniq/${run_names[$index]}"
    checkpoint="${output_root}/${run_names[$index]}/checkpoint_epoch_004.pt"
    if [[ ! -f "${fashion_output}/metrics.json" ]]; then
      "${PYTHON}" scripts/evaluate_fashioniq_clip.py \
        --model-name "${model_names[$index]}" \
        --config "${configs[$index]}" \
        --checkpoint "${checkpoint}" \
        --dataset-root "${fashioniq_root}" \
        --output-dir "${fashion_output}" \
        --device cuda:0
    fi
  done
else
  echo "[skip stages 4-6] RUN_EXTENDED_EVAL=${RUN_EXTENDED_EVAL}"
fi

echo "[stage 7/7] summarize experiment"
"${PYTHON}" scripts/summarize_opd_qminus_candidates.py \
  --output-root "${output_root}"

echo "[done] ${output_root}/RESULTS.md"
