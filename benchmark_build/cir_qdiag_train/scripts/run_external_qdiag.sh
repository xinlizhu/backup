#!/usr/bin/env bash
# Reproduce the two authorized external-model QDiag evaluations.
set -euo pipefail
source /mnt/data/datasets/search_dataset/zxl/for_conda/etc/profile.d/conda.sh
conda activate /mnt/data/datasets/search_dataset/zxl/multi_modal_py3.12
task_project=/mnt/data/datasets/search_dataset/zxl/benchmark_build/cir_qdiag_train
task_models=/mnt/data/datasets/search_dataset/zxl/models
task_output="${1:-${task_project}/outputs/external_qdiag_20261109}"
export HF_HOME="${task_models}/.hf_cache"
export TORCH_HOME="${task_models}/.torch_cache"
cd "${task_project}"
mkdir -p "${task_output}/TEMA-MCIRR" "${task_output}/CLIP4CIR-CIRR-RN50x4"
PYTHONPATH="${task_models}/qdiag_compat${PYTHONPATH:+:${PYTHONPATH}}" python -u scripts/evaluate_external_qdiag.py \
  --model tema --checkpoint "${task_models}/TEMA/TEMA_mcirr.pt" \
  --output "${task_output}/TEMA-MCIRR" \
  --image-batch-size 256 --query-batch-size 256 --workers 24 \
  >> "${task_output}/TEMA-MCIRR/runner.log" 2>&1
python -u scripts/evaluate_external_qdiag.py \
  --model clip4cir \
  --checkpoint "${task_models}/CLIP4CIR/weights/CIRR/RN50x4_fullft/cirr_comb_RN50x4_fullft.pt" \
  --clip-checkpoint "${task_models}/CLIP4CIR/weights/CIRR/RN50x4_fullft/cirr_clip_RN50x4_fullft.pt" \
  --output "${task_output}/CLIP4CIR-CIRR-RN50x4" \
  --image-batch-size 256 --query-batch-size 256 --workers 24 \
  >> "${task_output}/CLIP4CIR-CIRR-RN50x4/runner.log" 2>&1
