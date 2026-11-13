#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python}"
args=()
if [[ -n "${RUN_INDICES:-}" ]]; then
  read -r -a indices <<< "$RUN_INDICES"
  args+=(--run-indices "${indices[@]}")
fi
if [[ "${RUN_EXTENDED_EVAL:-1}" != "1" ]]; then args+=(--skip-extended-eval); fi
if [[ "${RUN_SUMMARY:-1}" != "1" ]]; then args+=(--skip-summary); fi
exec "$PYTHON" scripts/run_epoch_target_filter.py "${args[@]}" "$@"
