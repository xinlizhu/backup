#!/usr/bin/env bash
set -euo pipefail

# Six GPUs, one independent single-GPU experiment per device.
# Example: GPU_IDS=0,1,2,3,4,5 bash scripts/run_clip_batch64_weights_parallel.sh
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5}"
IFS=',' read -r -a gpu_ids <<< "${GPU_IDS}"
experiments=(
  clip_b:baseline
  clip_b:hard_w0_5
  clip_b:hard_w1_0
  clip_l:baseline
  clip_l:hard_w0_5
  clip_l:hard_w1_0
)

if [[ "${#gpu_ids[@]}" -lt "${#experiments[@]}" ]]; then
  echo "Need six GPU IDs; got ${#gpu_ids[@]} in GPU_IDS=${GPU_IDS}" >&2
  exit 2
fi

run_one() {
  local gpu="$1"
  local specification="$2"
  local variant="${specification%%:*}"
  local mode="${specification##*:}"
  local config="configs/${variant}_b64_${mode}.yaml"
  local output="outputs/${variant}/batch64/${mode}"
  mkdir -p "${output}"
  CUDA_VISIBLE_DEVICES="${gpu}" torchrun --standalone --nproc_per_node=1 \
    scripts/train.py --config "${config}" >"${output}/run.log" 2>&1
  CUDA_VISIBLE_DEVICES="${gpu}" python scripts/evaluate.py \
    --config "${config}" --checkpoint "${output}/last.pt" \
    >>"${output}/run.log" 2>&1
}

pids=()
for index in "${!experiments[@]}"; do
  run_one "${gpu_ids[index]}" "${experiments[index]}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  echo "At least one experiment failed; inspect outputs/*/batch64/*/run.log" >&2
  exit "${status}"
fi

for variant in clip_b clip_l; do
  python scripts/compare_hard_weights.py \
    --baseline "outputs/${variant}/batch64/baseline/val_metrics.json" \
    --weight-0-5 "outputs/${variant}/batch64/hard_w0_5/val_metrics.json" \
    --weight-1-0 "outputs/${variant}/batch64/hard_w1_0/val_metrics.json" \
    --output "outputs/${variant}/batch64/comparison.json"
done
