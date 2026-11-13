#!/usr/bin/env bash
set -euo pipefail

# Evaluate existing Full checkpoints at epochs 2/4/6/8/10. Override RUNS to
# select a subset. Missing runs are rejected instead of silently ignored.
RUNS="${RUNS:-baseline hard_w0_1 hard_w0_25 hard_w0_5}"

config_for_run() {
  case "$1" in
    baseline) echo "configs/clip_l_full_baseline.yaml" ;;
    hard_w0_1) echo "configs/clip_l_full_hard_w0_1.yaml" ;;
    hard_w0_25) echo "configs/clip_l_full_hard_w0_25.yaml" ;;
    hard_w0_5) echo "configs/clip_l_full_hard_w0_5.yaml" ;;
    hard_w0_75) echo "configs/clip_l_full_hard_w0_75.yaml" ;;
    hard_w1_0) echo "configs/clip_l_full_hard_w1_0.yaml" ;;
    *)
      echo "Unsupported run: $1" >&2
      return 2
      ;;
  esac
}

for run in ${RUNS}; do
  config="$(config_for_run "${run}")"
  python scripts/evaluate_checkpoint_curve.py --config "${config}"
done
