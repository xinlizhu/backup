#!/usr/bin/env bash
set -euo pipefail
source /mnt/data/datasets/search_dataset/zxl/for_conda/etc/profile.d/conda.sh
task_project=/mnt/data/datasets/search_dataset/zxl/benchmark_build/cir_qdiag_train
task_output="${1:-${task_project}/outputs/external_qdiag_20261109}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "${task_project}"
for task_size in base large; do
  if [[ "${task_size}" == base ]]; then
    task_name=MagicLens-CLIP-Base
    task_image_batch=256
  else
    task_name=MagicLens-CLIP-Large
    task_image_batch=128
  fi
  mkdir -p "${task_output}/${task_name}"
  conda activate /mnt/data/datasets/search_dataset/zxl/magic_lens311
  python -u scripts/export_magiclens_qdiag.py --size "${task_size}" \
    --output "${task_output}/${task_name}" --image-batch-size "${task_image_batch}" \
    --text-batch-size 512 >> "${task_output}/${task_name}/runner.log" 2>&1
  conda activate /mnt/data/datasets/search_dataset/zxl/multi_modal_py3.12
  python -u scripts/summarize_magiclens_qdiag.py --output "${task_output}/${task_name}" \
    >> "${task_output}/${task_name}/runner.log" 2>&1
done
python scripts/collect_external_qdiag_results.py --root "${task_output}"
