#!/usr/bin/env bash
set -euo pipefail
source /mnt/data/datasets/search_dataset/zxl/for_conda/etc/profile.d/conda.sh
task_project=/mnt/data/datasets/search_dataset/zxl/benchmark_build/cir_qdiag_train
task_models=/mnt/data/datasets/search_dataset/zxl/models
task_output="${1:-${task_project}/outputs/external_retrieval_20261109}"
export HF_HOME="${task_models}/.hf_cache"
export TORCH_HOME="${task_models}/.torch_cache"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "${task_project}"
conda activate /mnt/data/datasets/search_dataset/zxl/multi_modal_py3.12
python scripts/prepare_external_retrieval.py --output "${task_output}/manifests"
mkdir -p "${task_output}/TEMA-MCIRR" "${task_output}/CLIP4CIR-CIRR-RN50x4"
PYTHONPATH="${task_models}/qdiag_compat${PYTHONPATH:+:${PYTHONPATH}}" python -u scripts/evaluate_external_retrieval.py \
 --model tema --model-name TEMA-MCIRR --checkpoint "${task_models}/TEMA/TEMA_mcirr.pt" \
 --output "${task_output}/TEMA-MCIRR" --manifests "${task_output}/manifests" \
 --image-batch-size 256 --query-batch-size 256 --workers 16 >> "${task_output}/TEMA-MCIRR/runner.log" 2>&1
python -u scripts/evaluate_external_retrieval.py --model clip4cir --model-name CLIP4CIR-CIRR-RN50x4 \
 --checkpoint "${task_models}/CLIP4CIR/weights/CIRR/RN50x4_fullft/cirr_comb_RN50x4_fullft.pt" \
 --clip-checkpoint "${task_models}/CLIP4CIR/weights/CIRR/RN50x4_fullft/cirr_clip_RN50x4_fullft.pt" \
 --output "${task_output}/CLIP4CIR-CIRR-RN50x4" --manifests "${task_output}/manifests" \
 --image-batch-size 256 --query-batch-size 256 --workers 16 >> "${task_output}/CLIP4CIR-CIRR-RN50x4/runner.log" 2>&1
conda activate /mnt/data/datasets/search_dataset/zxl/magic_lens311
for task_size in base large; do
 if [[ "${task_size}" == base ]]; then task_name=MagicLens-CLIP-Base;task_batch=256;else task_name=MagicLens-CLIP-Large;task_batch=128;fi
 mkdir -p "${task_output}/${task_name}"
 python -u scripts/evaluate_magiclens_retrieval.py --size "${task_size}" --output "${task_output}/${task_name}" \
  --manifests "${task_output}/manifests" --image-batch-size "${task_batch}" --text-batch-size 512 --workers 12 \
  >> "${task_output}/${task_name}/runner.log" 2>&1
done
conda activate /mnt/data/datasets/search_dataset/zxl/multi_modal_py3.12
python scripts/collect_external_retrieval_results.py --root "${task_output}" --verify-overlap
