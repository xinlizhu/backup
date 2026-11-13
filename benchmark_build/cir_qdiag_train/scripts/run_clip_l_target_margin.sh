#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
benchmark_root="$(dirname "${project_root}")"
python_bin="${PYTHON:-/mnt/data/datasets/search_dataset/zxl/multi_modal_py3.12/bin/python}"
initialize_from="${project_root}/outputs/clip_l/full/hard_w0_1/checkpoint_epoch_010.pt"

cd "${project_root}"
if [[ ! -f "${initialize_from}" ]]; then
  echo "Missing initialization checkpoint: ${initialize_from}" >&2
  exit 1
fi

for run in ${RUNS:-control margin0_05}; do
  case "${run}" in
    control)
      config="configs/clip_l_full_hard_w0_1_continue_v5.yaml"
      output_dir="outputs/clip_l/full/hard_w0_1_continue_v5"
      ;;
    margin0_05)
      config="configs/clip_l_full_target_margin0_05_continue_v5.yaml"
      output_dir="outputs/clip_l/full/target_margin0_05_continue_v5"
      ;;
    *)
      echo "Unsupported run: ${run}; expected control or margin0_05" >&2
      exit 2
      ;;
  esac

  final_checkpoint="${output_dir}/checkpoint_epoch_003.pt"
  if [[ -f "${final_checkpoint}" ]]; then
    echo "[skip train] ${final_checkpoint}"
  elif [[ -f "${output_dir}/last.pt" ]]; then
    echo "[resume train] ${output_dir}/last.pt"
    "${python_bin}" -m torch.distributed.run --standalone --nproc_per_node=1 \
      scripts/train.py \
      --config "${config}" \
      --resume-from "${output_dir}/last.pt"
  else
    echo "[train] target-conditioned hard ${run}"
    "${python_bin}" -m torch.distributed.run --standalone --nproc_per_node=1 \
      scripts/train.py \
      --config "${config}" \
      --initialize-from "${initialize_from}"
  fi

  "${python_bin}" scripts/evaluate.py \
    --config "${config}" \
    --checkpoint "${final_checkpoint}" \
    --output "${output_dir}/val_metrics.json"
  "${python_bin}" scripts/evaluate_reverse.py \
    --config "${config}" \
    --checkpoint "${final_checkpoint}" \
    --output "${output_dir}/reverse_val_metrics.json"
done

if [[ "${RUN_PINPOINT:-1}" == "1" ]]; then
  (
    cd "${benchmark_root}/diag_model"
    "${python_bin}" run_clip_pinpoint_official.py \
      --models "CLIP-L-target-margin0.05" \
      --checkpoint "CLIP-L-target-margin0.05=${project_root}/outputs/clip_l/full/target_margin0_05_continue_v5/checkpoint_epoch_003.pt" \
      --output-root "${project_root}/outputs/target_margin_pinpoint_official" \
      --batch-size "${PINPOINT_BATCH_SIZE:-64}"
  )
fi
