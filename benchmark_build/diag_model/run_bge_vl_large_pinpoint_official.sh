#!/usr/bin/env bash
set -euo pipefail

python run_clip_pinpoint_official.py \
  --models BGE-VL-large-original,BGE-VL-large-full-hard-w0.1 \
  --output-root ../cir_qdiag_openai/outputs/pinpoint_official_bge_vl_large \
  --batch-size "${BATCH_SIZE:-64}"
