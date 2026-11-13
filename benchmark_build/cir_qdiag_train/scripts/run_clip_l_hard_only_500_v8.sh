#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
benchmark_root="$(dirname "${project_root}")"
python_bin="${PYTHON_BIN:-/mnt/data/datasets/search_dataset/zxl/multi_modal_py3.12/bin/python}"
config="${project_root}/configs/clip_l_full_baseline_then_hard_only_500_v8.yaml"
baseline_config="${project_root}/configs/clip_l_full_baseline.yaml"
baseline_checkpoint="${project_root}/outputs/clip_l/full/baseline/checkpoint_epoch_010.pt"
run_dir="${project_root}/outputs/clip_l/full/baseline_then_hard_only_500_v8"
manifest="${project_root}/data/cirr_train_pilot_500_v8.jsonl"
negative_jsonl="${benchmark_root}/cir_qdiag_openai_cirr/outputs/cirr_train_pilot_500_v8/final.jsonl"
cirr_root="/mnt/data/datasets/search_dataset/zxl/CIR_new/datasets/CIRR"
selected_epochs="${HARD_EPOCHS:-2,6,10}"

if [[ ! -f "${baseline_checkpoint}" ]]; then
  echo "Missing normal Full CLIP-L baseline checkpoint: ${baseline_checkpoint}" >&2
  exit 1
fi

"${python_bin}" "${project_root}/scripts/prepare_train_manifest.py" \
  --annotations "${cirr_root}/cirr/captions/cap.rc2.train.json" \
  --image-split "${cirr_root}/cirr/image_splits/split.rc2.train.json" \
  --image-root "${cirr_root}" \
  --negative-jsonl "${negative_jsonl}" \
  --only-samples-with-negatives \
  --output "${manifest}"

manifest_rows="$(wc -l < "${manifest}")"
if [[ "${manifest_rows}" -ne 500 ]]; then
  echo "Expected exactly 500 training samples, found ${manifest_rows}: ${manifest}" >&2
  exit 1
fi

final_checkpoint="${run_dir}/checkpoint_epoch_010.pt"
if [[ ! -f "${final_checkpoint}" ]]; then
  if [[ -f "${run_dir}/last.pt" ]]; then
    "${python_bin}" -m torch.distributed.run --standalone --nproc_per_node="${GPUS:-1}" \
      "${project_root}/scripts/train.py" \
      --config "${config}" \
      --resume-from "${run_dir}/last.pt"
  else
    "${python_bin}" -m torch.distributed.run --standalone --nproc_per_node="${GPUS:-1}" \
      "${project_root}/scripts/train.py" \
      --config "${config}" \
      --initialize-from "${baseline_checkpoint}"
  fi
else
  echo "[skip] training complete: ${final_checkpoint}"
fi

comparison_root="${run_dir}/comparison"
"${python_bin}" "${project_root}/scripts/evaluate_hard_only_500_experiment.py" \
  --config "${config}" \
  --baseline-config "${baseline_config}" \
  --baseline-checkpoint "${baseline_checkpoint}" \
  --run-dir "${run_dir}" \
  --epochs "${selected_epochs}" \
  --output-dir "${comparison_root}/cirr_val"

if [[ "${RUN_PINPOINT:-1}" == "1" ]]; then
  IFS=',' read -r -a epochs <<< "${selected_epochs}"
  model_names=("CLIP-L-original" "CLIP-L-full-baseline")
  checkpoint_args=()
  for raw_epoch in "${epochs[@]}"; do
    epoch="$((10#${raw_epoch}))"
    model_name="CLIP-L-hard-only-500-v8-e$(printf '%03d' "${epoch}")"
    checkpoint_path="${run_dir}/checkpoint_epoch_$(printf '%03d' "${epoch}").pt"
    if [[ ! -f "${checkpoint_path}" ]]; then
      echo "Missing selected hard-only checkpoint: ${checkpoint_path}" >&2
      exit 1
    fi
    model_names+=("${model_name}")
    checkpoint_args+=(--checkpoint "${model_name}=${checkpoint_path}")
  done
  models_csv="$(IFS=,; echo "${model_names[*]}")"
  (
    cd "${benchmark_root}/diag_model"
    "${python_bin}" run_clip_pinpoint_official.py \
      --models "${models_csv}" \
      "${checkpoint_args[@]}" \
      --output-root "${comparison_root}/pinpoint_official" \
      --batch-size "${PINPOINT_BATCH_SIZE:-64}"
  )
fi

echo "[done] CIRR: ${comparison_root}/cirr_val/cirr_val_comparison.csv"
if [[ "${RUN_PINPOINT:-1}" == "1" ]]; then
  echo "[done] PinPoint: ${comparison_root}/pinpoint_official/official_map10_negrecall10.csv"
fi
