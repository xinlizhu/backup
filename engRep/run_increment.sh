#!/usr/bin/env bash
# 处理增量：bash run_increment.sh --start-date 2026-07-01 --batch-date 2026_10_13
# --batch-date 可省略，默认北京时间当天；--workers 默认 8 个报告并行。
# --max-reports 2 可先处理两篇；--reuse-server 可复用端口上的同名模型服务。
# 默认自动启动当前 PYTHON_BIN 同环境下的 vLLM，并在退出时关闭自己启动的服务。
# 默认端口 7888，可通过 PORT 环境变量或 --port 覆盖。
# 默认 GPU 显存利用率 0.85，可通过 GPU_MEMORY_UTILIZATION 或 --gpu-memory-utilization 覆盖。
# 抽取服务退出后自动运行 Qwen3-VL-Embedding-2B，只编码待发布图表，默认显存预算 0.95。
# --embedding-gpu-memory-utilization / --embedding-batch-size 可单独调整向量阶段。
# --reuse-server 不会关闭外部抽取服务，需自行保证向量阶段的可用显存。
# 存储：data/picture/YYYYMMDD/ 放图片，data/jsonl/YYYYMMDD.jsonl 为每批索引。
# data/full.jsonl 为全量图表索引；两种索引的图片路径均相对于各自文件目录。
# 问句、运行记录、日志及续跑缓存全部存放到相邻的 data_mid/。
# 仅剩 upload_data 时可从发布包恢复历史索引，历史图片继续引用 upload_data/pic/。
# 收尾后生成带 data.embedding（2048 维、L2 归一化）的新发布批次；历史批次不重写。
# upload_data/：pic/YYYYMMDD/ 图片及 research-report-images/ 协议包。
# ID 前缀 ResearchReportImages:；日期批次 YYYYMMDD000000000-0000，同日新增递增末尾序号。
# 每个 gzip <=10 MiB，不再生成未压缩 JSONL 预览；Meta 保留 180 天。
# embedding 失败不发布 Manifest/Meta；缓存留在 data_mid/embedding，重跑可补齐。
# data_mid/YYYY_MM_DD/publication.json 单独记录发布阶段状态。
# 仅本地打包，不上传 OSS；--upload-dir 可指定打包目录。
# 图片路径相对于各自 JSONL 所在目录。
# 中断：Ctrl+C；当前任务收尾后退出，再运行同一命令即可续跑。
# 失败报告保留缓存，后续运行会重试；无合格图表的成功报告不会重复处理。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
START_DATE="${START_DATE:-2026-06-01}"
export PYTHONDONTWRITEBYTECODE=1
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost"
exec "$PYTHON_BIN" -u "$SCRIPT_DIR/incremental_reports.py" run --start-date "$START_DATE" "$@"
