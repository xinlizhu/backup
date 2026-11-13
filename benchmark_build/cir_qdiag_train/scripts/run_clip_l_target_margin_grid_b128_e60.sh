#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
benchmark_root="$(dirname "${project_root}")"
python_bin="${PYTHON:-/mnt/data/datasets/search_dataset/zxl/multi_modal_py3.12/bin/python}"
config="${project_root}/configs/clip_l_full_target_margin_grid_b128_e60.yaml"
initialize_from="${project_root}/outputs/clip_l/full/hard_w0_1/checkpoint_epoch_010.pt"
grid_root="${project_root}/outputs/clip_l/full/target_margin_grid_b128_e60"
timing_file="${grid_root}/timing.tsv"

mkdir -p "${grid_root}"
cd "${project_root}"
if [[ ! -f "${initialize_from}" ]]; then
  echo "Missing initialization checkpoint: ${initialize_from}" >&2
  exit 1
fi
if [[ ! -f "${timing_file}" ]]; then
  printf 'phase\trun\tstart_utc\tend_utc\tseconds\tstatus\n' > "${timing_file}"
fi

record_timing() {
  local phase="$1"
  local run="$2"
  local start_utc="$3"
  local end_utc="$4"
  local seconds="$5"
  local status="$6"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${phase}" "${run}" "${start_utc}" "${end_utc}" "${seconds}" "${status}" \
    >> "${timing_file}"
}

run_timed() {
  local phase="$1"
  local run="$2"
  local log_path="$3"
  shift 3
  local start_utc
  local end_utc
  local start_seconds
  local end_seconds
  local exit_status
  start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  start_seconds="$(date +%s)"
  set +e
  "$@" 2>&1 | tee -a "${log_path}"
  exit_status="${PIPESTATUS[0]}"
  set -e
  end_seconds="$(date +%s)"
  end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [[ "${exit_status}" -eq 0 ]]; then
    record_timing "${phase}" "${run}" "${start_utc}" "${end_utc}" \
      "$((end_seconds - start_seconds))" "ok"
  else
    record_timing "${phase}" "${run}" "${start_utc}" "${end_utc}" \
      "$((end_seconds - start_seconds))" "failed:${exit_status}"
    return "${exit_status}"
  fi
}

margin_tag() {
  case "$1" in
    0.05) echo "0_05" ;;
    0.1) echo "0_1" ;;
    0.2) echo "0_2" ;;
    *) return 2 ;;
  esac
}

weight_tag() {
  case "$1" in
    0.1) echo "0_1" ;;
    0.5) echo "0_5" ;;
    *) return 2 ;;
  esac
}

grid_start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
grid_start_seconds="$(date +%s)"
model_names=()
checkpoint_args=()

for margin in 0.05 0.1 0.2; do
  for weight in 0.1 0.5; do
    run="m$(margin_tag "${margin}")_w$(weight_tag "${weight}")"
    output_dir="${grid_root}/${run}"
    final_checkpoint="${output_dir}/checkpoint_epoch_060.pt"
    mkdir -p "${output_dir}"

    train_arguments=(
      "${python_bin}" -m torch.distributed.run --standalone --nproc_per_node=1
      scripts/train.py
      --config "${config}"
      --experiment-name "cirr_clip_l_full_${run}_b128_e60"
      --output-dir "${output_dir}"
      --epochs 60
      --batch-size-per-gpu 128
      --gradient-accumulation-steps 1
      --hard-negative-weight "${weight}"
      --hard-negative-margin "${margin}"
      --save-every-epochs 20
    )
    if [[ -f "${final_checkpoint}" ]]; then
      echo "[skip train] ${final_checkpoint}"
      now_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      record_timing "train" "${run}" "${now_utc}" "${now_utc}" 0 "skipped"
    elif [[ -f "${output_dir}/last.pt" ]]; then
      echo "[resume train] ${run}"
      run_timed "train" "${run}" "${output_dir}/train_console.log" \
        "${train_arguments[@]}" --resume-from "${output_dir}/last.pt"
    else
      echo "[train] ${run}: margin=${margin}, hard_weight=${weight}"
      run_timed "train" "${run}" "${output_dir}/train_console.log" \
        "${train_arguments[@]}" --initialize-from "${initialize_from}"
    fi

    for epoch in 020 040 060; do
      checkpoint="${output_dir}/checkpoint_epoch_${epoch}.pt"
      if [[ ! -f "${checkpoint}" ]]; then
        echo "Missing scheduled checkpoint: ${checkpoint}" >&2
        exit 1
      fi
      run_timed "cirr_epoch_${epoch}" "${run}" \
        "${output_dir}/eval_epoch_${epoch}.log" \
        "${python_bin}" scripts/evaluate.py \
          --config "${config}" \
          --checkpoint "${checkpoint}" \
          --output "${output_dir}/val_metrics_epoch_${epoch}.json"
    done

    run_timed "reverse_epoch_060" "${run}" \
      "${output_dir}/reverse_eval.log" \
      "${python_bin}" scripts/evaluate_reverse.py \
        --config "${config}" \
        --checkpoint "${final_checkpoint}" \
        --output "${output_dir}/reverse_val_metrics.json"

    model_name="CLIP-L-margin-${run}-e60"
    model_names+=("${model_name}")
    checkpoint_args+=(--checkpoint "${model_name}=${final_checkpoint}")
  done
done

if [[ "${RUN_PINPOINT:-1}" == "1" ]]; then
  models_csv="$(IFS=,; echo "${model_names[*]}")"
  run_timed "pinpoint_all" "all" "${grid_root}/pinpoint_eval.log" \
    bash -c 'cd "$1" && shift && exec "$@"' _ \
      "${benchmark_root}/diag_model" \
      "${python_bin}" run_clip_pinpoint_official.py \
        --models "${models_csv}" \
        "${checkpoint_args[@]}" \
        --output-root "${grid_root}/pinpoint_official" \
        --batch-size "${PINPOINT_BATCH_SIZE:-64}"
fi

grid_end_seconds="$(date +%s)"
grid_end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
record_timing "total" "all" "${grid_start_utc}" "${grid_end_utc}" \
  "$((grid_end_seconds - grid_start_seconds))" "ok"

"${python_bin}" scripts/summarize_target_margin_grid.py --root "${grid_root}"
echo "[done] ${grid_root}/grid_summary.csv"
echo "[timing] ${timing_file}"
