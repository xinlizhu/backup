#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_ENV="${PYTHON_ENV:-/mnt/data/datasets/search_dataset/zxl/multi_modal_py3.12}"
cd "$PROJECT_ROOT"

"$PYTHON_ENV/bin/python" run_clip_pinpoint_official.py \
  --models CLIP-L-OAP-epoch2-beta0.1,CLIP-L-OAP-epoch5-beta0.1,BGE-VL-large-OAP-epoch5-beta0.025 \
  --output-root ../cir_qdiag_openai/outputs/pinpoint_official_oap \
  --batch-size "${BATCH_SIZE:-64}"

"$PYTHON_ENV/bin/python" summarize_oap_pinpoint.py \
  --oap-summary ../cir_qdiag_openai/outputs/pinpoint_official_oap/official_map10_negrecall10.csv \
  --clip-baseline ../cir_qdiag_openai/outputs/pinpoint_official_full_lora/official_map10_negrecall10.csv \
  --bge-baseline ../cir_qdiag_openai/outputs/pinpoint_official_bge_vl_large/official_map10_negrecall10.csv \
  --output ../cir_qdiag_openai/outputs/pinpoint_official_oap/oap_vs_global.csv
