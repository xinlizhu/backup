#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
read -r -a indices <<< "${RUN_INDICES:-0 1 2 3 4}"
args=(--run-indices "${indices[@]}")
if [[ "${RUN_EXTENDED_EVAL:-1}" == "0" ]]; then
  args+=(--skip-extended-eval)
fi
if [[ "${RUN_SUMMARY:-1}" == "0" ]]; then
  args+=(--skip-summary)
fi
cd "${project_root}"
exec "${PYTHON}" scripts/run_reverse_rank_r128.py "${args[@]}" "$@"
