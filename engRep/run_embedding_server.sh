#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}/models/Qwen3-VL-Embedding-2B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-search-report-image-embedding}"
PORT="${PORT:-8088}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.09}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2000}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
VLLM_BIN="${VLLM_BIN:-vllm}"
export NVIDIA_DISABLE_REQUIRE="${NVIDIA_DISABLE_REQUIRE:-1}"
export PYTHONDONTWRITEBYTECODE=1

args=(serve "$MODEL_PATH"
  --served-model-name "$SERVED_MODEL_NAME"
  --dtype bfloat16
  --max-model-len "$MAX_MODEL_LEN"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --trust-remote-code
  --max-num-seqs "$MAX_NUM_SEQS"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --enable-prefix-caching
  --runner pooling
  --convert embed
  --mm-processor-cache-type shm
  --limit-mm-per-prompt '{"image":1,"video":0}'
  --host 0.0.0.0 --port "$PORT")

# Optional explicit token budget; otherwise retain the installed vLLM default.
if [[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ]]; then
  args+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
fi
args+=("$@")
printf 'Launching:'
printf ' %q' "$VLLM_BIN" "${args[@]}"
printf '\n'
if [[ "${DRY_RUN:-0}" == 1 ]]; then exit 0; fi
exec "$VLLM_BIN" "${args[@]}"
