#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}/models/qwen3.8-27b}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-27b}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
PORT="${PORT:-8088}"
EXTRA_ARGS=()
if [[ "${LIMIT_MM_PER_PROMPT+x}" != x ]]; then
  LIMIT_MM_PER_PROMPT='{"image": 2}'
fi
if [[ -n "$LIMIT_MM_PER_PROMPT" ]]; then
  EXTRA_ARGS+=(--limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT")
fi
# 后台负载脚本可选择 eager 模式，避开 CUDA Graph；普通服务默认保持原行为。
if [[ "${ENFORCE_EAGER:-0}" == 1 ]]; then
  EXTRA_ARGS+=(--enforce-eager)
fi
if [[ -n "${MAX_NUM_SEQS:-}" ]]; then
  EXTRA_ARGS+=(--max-num-seqs "$MAX_NUM_SEQS")
fi
VLLM_BIN="${VLLM_BIN:-vllm}"

exec "$VLLM_BIN" serve "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --dtype bfloat16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port "$PORT" \
  "${EXTRA_ARGS[@]}"
