#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Simple no-ML analysis: Full comparison, LoRA replication, frozen-image control.
# Models are processed sequentially; six checkpoints never coexist on GPU.
# --skip-probes explicitly disables OOF LogisticRegression and learned reranking.
python counterfactual_profile_analysis.py \
  --models CLIP-L-full-baseline,CLIP-L-full-hard-w0.5,CLIP-L-lora-r16-baseline,CLIP-L-lora-r16-hard-w0.5,CLIP-L-baseline,CLIP-L-hard-w0.5 \
  --comparison CLIP-L-full-baseline,CLIP-L-full-hard-w0.5 \
  --comparison CLIP-L-lora-r16-baseline,CLIP-L-lora-r16-hard-w0.5 \
  --comparison CLIP-L-baseline,CLIP-L-hard-w0.5 \
  --device "${DEVICE:-cuda:0}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --num-workers "${NUM_WORKERS:-16}" \
  --bootstrap-replicates "${BOOTSTRAP_REPLICATES:-2000}" \
  --skip-probes \
  "$@"
