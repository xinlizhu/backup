#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
benchmark_root="$(dirname "${project_root}")"
python_bin="${PYTHON:-/mnt/data/datasets/search_dataset/zxl/multi_modal_py3.12/bin/python}"
grid_root="${project_root}/outputs/clip_l/full/target_margin_grid_b128_e60"
output_root="${grid_root}/winrate_random10"

cd "${benchmark_root}/diag_model"
"${python_bin}" evaluate_winrate_random10.py \
  --checkpoint "m0_05_w0_1_e60=${grid_root}/m0_05_w0_1/checkpoint_epoch_060.pt" \
  --checkpoint "m0_05_w0_5_e60=${grid_root}/m0_05_w0_5/checkpoint_epoch_060.pt" \
  --checkpoint "m0_1_w0_1_e60=${grid_root}/m0_1_w0_1/checkpoint_epoch_060.pt" \
  --output-root "${output_root}" \
  --device cuda:0 \
  --batch-size "${DIAG_BATCH_SIZE:-64}" \
  --random-samples 10 \
  --bootstrap-replicates 2000 \
  --seed 42

echo "[done] ${output_root}/overview_all_strict.csv"
