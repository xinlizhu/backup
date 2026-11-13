#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
benchmark_root="$(dirname "${project_root}")"
workspace_root="$(dirname "${benchmark_root}")"
output_root="${OUTPUT_ROOT:-${project_root}/outputs/qwen3_vl_original_baselines}"
device="${DEVICE:-cuda:0}"
dtype="${DTYPE:-auto}"

model_names=(
  Qwen3-VL-Embedding-2B
  Qwen3-VL-Embedding-8B
)
model_paths=(
  "${workspace_root}/CIR_PLAY/models/Qwen3-VL-Embedding-2B"
  "${workspace_root}/CIR_PLAY/models/Qwen3-VL-Embedding-8B"
)
batch_sizes=(
  "${QWEN2B_BATCH_SIZE:-16}"
  "${QWEN8B_BATCH_SIZE:-4}"
)

for index in "${!model_names[@]}"; do
  if [[ ! -f "${model_paths[$index]}/config.json" ]]; then
    echo "[error] missing model: ${model_paths[$index]}" >&2
    exit 1
  fi
done

mkdir -p "${output_root}"
cd "${project_root}"

echo "[stage 1/5] CIRR val: original Qwen3-VL-Embedding 2B and 8B"
for index in "${!model_names[@]}"; do
  model_name="${model_names[$index]}"
  model_path="${model_paths[$index]}"
  batch_size="${batch_sizes[$index]}"
  model_output="${output_root}/cirr/${model_name}"
  if [[ -f "${model_output}/val_metrics.json" ]]; then
    echo "[skip] ${model_name} CIRR"
    continue
  fi
  "${PYTHON}" scripts/evaluate_cirr_original_adapter.py \
    --model-name "${model_name}" \
    --model-path "${model_path}" \
    --output-dir "${model_output}" \
    --device "${device}" \
    --dtype "${dtype}" \
    --image-batch-size "${batch_size}" \
    --query-batch-size "${batch_size}" \
    --score-batch-size "${SCORE_BATCH_SIZE:-512}"
done

echo "[stage 2/5] CIR-QDiag: Delta, Raw/Stable Win, random-10"
for index in "${!model_names[@]}"; do
  model_name="${model_names[$index]}"
  model_path="${model_paths[$index]}"
  batch_size="${batch_sizes[$index]}"
  if [[ -f "${output_root}/qdiag/${model_name}/summary.csv" ]]; then
    echo "[skip] ${model_name} QDiag"
    continue
  fi
  (
    cd "${benchmark_root}/diag_model"
    "${PYTHON}" evaluate_winrate_random10.py \
      --pretrained-model "${model_name}=${model_path}" \
      --output-root "${output_root}/qdiag" \
      --device "${device}" \
      --dtype "${dtype}" \
      --batch-size "${batch_size}" \
      --num-workers "${NUM_WORKERS:-32}" \
      --processor-workers "${PROCESSOR_WORKERS:-1}" \
      --random-samples 10 \
      --bootstrap-replicates "${QDIAG_BOOTSTRAP_REPLICATES:-2000}" \
      --seed 42
  )
done

echo "[stage 3/5] FashionIQ val: cross-dataset retrieval"
for index in "${!model_names[@]}"; do
  model_name="${model_names[$index]}"
  model_path="${model_paths[$index]}"
  batch_size="${batch_sizes[$index]}"
  model_output="${output_root}/fashioniq/${model_name}"
  if [[ -f "${model_output}/metrics.json" ]]; then
    echo "[skip] ${model_name} FashionIQ"
    continue
  fi
  "${PYTHON}" scripts/evaluate_fashioniq_original_adapter.py \
    --model-name "${model_name}" \
    --adapter-name "${model_name}" \
    --model-path "${model_path}" \
    --dataset-root "${FASHIONIQ_ROOT:-${workspace_root}/CIR_new/datasets/fashion-iq}" \
    --output-dir "${model_output}" \
    --device "${device}" \
    --dtype "${dtype}" \
    --image-batch-size "${batch_size}" \
    --query-batch-size "${batch_size}" \
    --score-batch-size "${SCORE_BATCH_SIZE:-1024}"
done

echo "[stage 4/5] write fast-suite summary before full PinPoint"
"${PYTHON}" scripts/summarize_qwen3_vl_original_baselines.py \
  --output-root "${output_root}"

if [[ "${RUN_PINPOINT_OFFICIAL:-1}" == "1" ]]; then
  echo "[stage 5/5] official PinPoint: full corpus"
  for index in "${!model_names[@]}"; do
    model_name="${model_names[$index]}"
    batch_size="${batch_sizes[$index]}"
    if [[ -f "${output_root}/pinpoint_official/${model_name}/official_metrics.csv" ]]; then
      echo "[skip] ${model_name} official PinPoint"
      continue
    fi
    (
      cd "${benchmark_root}/diag_model"
      "${PYTHON}" run_clip_pinpoint_official.py \
        --models "${model_name}" \
        --output-root "${output_root}/pinpoint_official" \
        --device "${device}" \
        --search-device "${device}" \
        --dtype "${dtype}" \
        --batch-size "${batch_size}" \
        --num-workers "${NUM_WORKERS:-32}" \
        --processor-workers "${PROCESSOR_WORKERS:-1}" \
        --query-search-batch-size "${SEARCH_QUERY_BATCH_SIZE:-256}" \
        --corpus-search-chunk-size "${SEARCH_CORPUS_CHUNK_SIZE:-8192}"
    )
  done
else
  echo "[skip] official PinPoint because RUN_PINPOINT_OFFICIAL=${RUN_PINPOINT_OFFICIAL}"
fi

"${PYTHON}" scripts/summarize_qwen3_vl_original_baselines.py \
  --output-root "${output_root}"

echo "[done] ${output_root}/QWEN3_VL_ORIGINAL_RESULTS.md"
