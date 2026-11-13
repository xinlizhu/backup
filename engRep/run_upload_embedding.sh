#!/usr/bin/env bash
# 为尚未上传的日期批次补齐 embedding；缓存和原包备份在 data_mid/embedding。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
exec "$PYTHON_BIN" -u "$SCRIPT_DIR/embed_upload.py" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.95}" "$@"
