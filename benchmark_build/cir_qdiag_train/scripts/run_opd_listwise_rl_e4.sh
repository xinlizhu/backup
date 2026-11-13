#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "${project_root}/scripts/run_opd_listwise_rl_e2.sh" "$@"
