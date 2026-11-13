#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
benchmark_root="$(dirname "${project_root}")"
workspace_root="$(dirname "${benchmark_root}")"
reranker_config="${project_root}/configs/opd_clip_l_qwen3_vl_reranker_2b_rank.yaml"
reranker_reverse_config="${project_root}/configs/opd_clip_l_qwen3_vl_reranker_2b_reverse_rank.yaml"
s2_reverse_config="${project_root}/configs/opd_clip_l_s2_reverse_rank.yaml"
output_root="${project_root}/outputs/opd_reranker_teacher"
reranker_run_dir="${output_root}/clip_l_from_qwen3_vl_reranker_2b"
reranker_reverse_run_dir="${output_root}/clip_l_from_qwen3_vl_reranker_2b_reverse_kl"
s2_reverse_run_dir="${output_root}/clip_l_from_s2_reverse_kl"
reranker_checkpoint="${reranker_run_dir}/checkpoint_epoch_004.pt"
reranker_reverse_checkpoint="${reranker_reverse_run_dir}/checkpoint_epoch_004.pt"
s2_reverse_checkpoint="${s2_reverse_run_dir}/checkpoint_epoch_004.pt"
reranker_model_name="CLIP-L-OPD-Qwen3VL-Reranker2B"
reranker_reverse_model_name="CLIP-L-OPD-Qwen3VL-Reranker2B-ReverseKL"
s2_reverse_model_name="CLIP-L-OPD-S2-ReverseKL"
s2_cache="${project_root}/outputs/on_policy_centered/cache/bge_mllm_s2_cirr_train"
fashioniq_root="${FASHIONIQ_ROOT:-${workspace_root}/CIR_new/datasets/fashion-iq}"

reranker_url="${RERANKER_URL:-http://interactive-kefnm35z3lpc:8001/v1/score}"
reranker_health_url="${RERANKER_HEALTH_URL:-http://interactive-kefnm35z3lpc:8001/health}"
reranker_model="${RERANKER_MODEL:-qwen3-vl-reranker-2b}"
reranker_api_format="${RERANKER_API_FORMAT:-score}"
reranker_transport="${RERANKER_IMAGE_TRANSPORT:-data_url}"
reranker_concurrency="${RERANKER_CONCURRENCY:-8}"

cd "${project_root}"
mkdir -p "${output_root}"

echo "[stage 0/8] verify both Teacher sources"
curl -fsS --max-time 20 "${reranker_health_url}" >/dev/null
echo "[ok] ${reranker_url} model=${reranker_model} api=${reranker_api_format} transport=${reranker_transport} concurrency=${reranker_concurrency}"
if [[ ! -f "${s2_cache}/metadata.json" ]]; then
  echo "[error] missing frozen S2 Teacher cache: ${s2_cache}/metadata.json" >&2
  exit 1
fi
echo "[ok] frozen S2 cache: ${s2_cache}"

reranker_teacher_args=(
  --teacher-endpoint-url "${reranker_url}"
  --teacher-model-name "${reranker_model}"
  --teacher-api-format "${reranker_api_format}"
  --teacher-image-transport "${reranker_transport}"
  --teacher-max-concurrent-requests "${reranker_concurrency}"
)

echo "[stage 1/8] train CLIP-L from reranker Teacher with forward KL"
if [[ -f "${reranker_checkpoint}" ]]; then
  echo "[skip train] ${reranker_checkpoint}"
elif [[ -f "${reranker_run_dir}/last.pt" ]]; then
  "${PYTHON}" scripts/train_on_policy_distillation.py \
    --config "${reranker_config}" \
    "${reranker_teacher_args[@]}" \
    --resume-from "${reranker_run_dir}/last.pt"
else
  "${PYTHON}" scripts/train_on_policy_distillation.py \
    --config "${reranker_config}" \
    "${reranker_teacher_args[@]}"
fi

echo "[stage 2/8] train an independent CLIP-L from reranker Teacher with reverse KL"
if [[ -f "${reranker_reverse_checkpoint}" ]]; then
  echo "[skip train] ${reranker_reverse_checkpoint}"
elif [[ -f "${reranker_reverse_run_dir}/last.pt" ]]; then
  "${PYTHON}" scripts/train_on_policy_distillation.py \
    --config "${reranker_reverse_config}" \
    "${reranker_teacher_args[@]}" \
    --resume-from "${reranker_reverse_run_dir}/last.pt"
else
  "${PYTHON}" scripts/train_on_policy_distillation.py \
    --config "${reranker_reverse_config}" \
    "${reranker_teacher_args[@]}"
fi

echo "[stage 3/8] train an independent CLIP-L from S2 with reverse KL"
if [[ -f "${s2_reverse_checkpoint}" ]]; then
  echo "[skip train] ${s2_reverse_checkpoint}"
elif [[ -f "${s2_reverse_run_dir}/last.pt" ]]; then
  "${PYTHON}" scripts/train_on_policy_distillation.py \
    --config "${s2_reverse_config}" \
    --resume-from "${s2_reverse_run_dir}/last.pt"
else
  "${PYTHON}" scripts/train_on_policy_distillation.py \
    --config "${s2_reverse_config}"
fi

run_dirs=("${reranker_run_dir}" "${reranker_reverse_run_dir}" "${s2_reverse_run_dir}")
configs=("${reranker_config}" "${reranker_reverse_config}" "${s2_reverse_config}")
model_names=("${reranker_model_name}" "${reranker_reverse_model_name}" "${s2_reverse_model_name}")
checkpoints=("${reranker_checkpoint}" "${reranker_reverse_checkpoint}" "${s2_reverse_checkpoint}")

echo "[stage 4/8] evaluate all four CIRR checkpoints for all Students"
for index in "${!run_dirs[@]}"; do
  for epoch in 001 002 003 004; do
    epoch_checkpoint="${run_dirs[$index]}/checkpoint_epoch_${epoch}.pt"
    epoch_metrics="${run_dirs[$index]}/val_epoch_${epoch}.json"
    if [[ ! -f "${epoch_metrics}" ]]; then
      "${PYTHON}" scripts/evaluate_on_policy_distillation.py \
        --config "${configs[$index]}" \
        --checkpoint "${epoch_checkpoint}" \
        --output "${epoch_metrics}"
    fi
  done
done

if [[ "${RUN_EXTENDED_EVAL:-1}" == "1" ]]; then
  echo "[stage 5/8] evaluate official PinPoint mAP@10 and NegRecall@10"
  (
    cd "${benchmark_root}/diag_model"
    "${PYTHON}" run_clip_pinpoint_official.py \
      --models "${reranker_model_name},${reranker_reverse_model_name},${s2_reverse_model_name}" \
      --checkpoint "${reranker_model_name}=${reranker_checkpoint}" \
      --checkpoint "${reranker_reverse_model_name}=${reranker_reverse_checkpoint}" \
      --checkpoint "${s2_reverse_model_name}=${s2_reverse_checkpoint}" \
      --output-root "${output_root}/pinpoint_official" \
      --device cuda:0 \
      --batch-size "${PINPOINT_BATCH_SIZE:-128}"
  )

  echo "[stage 6/8] evaluate QDiag Delta, Stable Win, and random-10 separation"
  (
    cd "${benchmark_root}/diag_model"
    "${PYTHON}" evaluate_winrate_random10.py \
      --checkpoint "${reranker_model_name}=${reranker_checkpoint}" \
      --checkpoint "${reranker_reverse_model_name}=${reranker_reverse_checkpoint}" \
      --checkpoint "${s2_reverse_model_name}=${s2_reverse_checkpoint}" \
      --output-root "${output_root}/qdiag" \
      --device cuda:0 \
      --batch-size "${QDIAG_BATCH_SIZE:-128}" \
      --random-samples 10 \
      --seed 42
  )

  echo "[stage 7/8] evaluate FashionIQ transfer"
  for index in "${!run_dirs[@]}"; do
    fashion_output="${output_root}/fashioniq/$(basename "${run_dirs[$index]}")"
    if [[ ! -f "${fashion_output}/metrics.json" ]]; then
      "${PYTHON}" scripts/evaluate_fashioniq_clip.py \
        --model-name "${model_names[$index]}" \
        --config "${configs[$index]}" \
        --checkpoint "${checkpoints[$index]}" \
        --dataset-root "${fashioniq_root}" \
        --output-dir "${fashion_output}" \
        --device cuda:0
    fi
  done
else
  echo "[skip stages 5-7] RUN_EXTENDED_EVAL=${RUN_EXTENDED_EVAL}"
fi

echo "[stage 8/8] summarize all experiments"
"${PYTHON}" scripts/summarize_opd_reranker_teacher.py \
  --output-root "${output_root}"

echo "[done] ${output_root}/RESULTS.md"
