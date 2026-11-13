#!/usr/bin/env bash
# 只读统计：bash check_increment.sh --start-date 2026-07-01
# start-date / end-date 筛选研报发布日期（含当天），不会启动模型。
# pending = new（无处理记录）+ retry（此前失败）；ready 表示主 JSON 已落盘。
# 只统计上游报告；从 upload_data、现有索引和处理回执识别已完成报告，不依赖迁移标记。
# 不修改 data/upload_data，也不打包或上传 OSS。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
# Change this default, set START_DATE, or pass --start-date YYYY-MM-DD.
START_DATE="${START_DATE:-2026-06-01}"
export PYTHONDONTWRITEBYTECODE=1
exec "$PYTHON_BIN" -u "$SCRIPT_DIR/incremental_reports.py" check --start-date "$START_DATE" "$@"
