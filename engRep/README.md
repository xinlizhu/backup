# engRep

研报图表增量处理代码：检查增量、抽取图片与描述、生成检索问句、计算 embedding，并生成本地 OSS 协议包。

本目录只包含代码、提示词、协议文档和测试。没有研报、图片、向量、模型权重、数据库、运行日志或访问凭据。

## 环境与外部输入

使用 Linux、Python 3.12。CPU 侧依赖：

```bash
python -m pip install -r requirements.txt
```

GPU 流程需要 CUDA 环境和 vLLM。`requirements-gpu.txt` 按 vLLM 0.19.0 的依赖约束提供安装入口；CUDA/PyTorch 构建需适配实际机器。备份整理时运行环境的具体版本见 [环境说明](docs/environment.md)。

模型和研报必须在本地另行准备：

```bash
export PYTHON_BIN=python3
export REPORT_DB=/path/to/research_index.db
export REPORT_SOURCE_ROOT=/path/to/json_output
export MODEL_PATH=/path/to/extraction-model
export EMBEDDING_MODEL_PATH=/path/to/Qwen3-VL-Embedding-2B
```

上游 SQLite 数据库包含 `research_reports` 表，检查流程使用 `pdf_id`、`json_relative_path`、`raw_date` 三个字段。`json_relative_path` 相对于 `REPORT_SOURCE_ROOT`，报告主 JSON 的父目录名应为报告 ID。抽取流程还需要上游生成的图片、版面信息，以及相应 PDF（用于可用时重新裁图）。

未设置环境变量时，默认寻找本目录下的 `external_data/` 和 `models/`；这些目录不会进入 Git。

## 常用命令

在 `engRep` 目录执行：

```bash
# 只读统计新增、此前失败、源文件缺失的报告。
bash check_increment.sh --start-date 2026-06-01

# 完整增量流程：抽取 → 释放抽取模型 → embedding → 新批次打包。
bash run_increment.sh --start-date 2026-06-01
```

`--start-date` 筛选报告发布日期（含当天），不是处理批次日期。`--end-date` 可限制截止日期；`--batch-date YYYYMMDD` 可指定批次日期，否则使用北京时间当天。

- 抽取：默认端口 `7888`、8 篇报告并行、GPU 显存预算 `0.85`。
- embedding：默认使用 Qwen3-VL-Embedding-2B，GPU 显存预算 `0.95`；通过 `--embedding-gpu-memory-utilization` 和 `--embedding-batch-size` 单独调整。
- `--max-reports 2` 可先处理两篇。`--reuse-server` 复用外部抽取服务，脚本不会关闭该服务；须为 embedding 阶段安排足够显存。
- `Ctrl+C` 后保留已完成报告和向量缓存，下次继续。
- 只补算并发布已抽取、尚未发布的数据：`python prepare_upload.py --batch-date YYYYMMDD`。
- 所有命令只生成本地文件，不会自动上传 OSS。

## 输出位置（运行时生成）

```text
data/                         本地图表索引、全量索引和新增图片
data_mid/                     抽取日志、问句、处理回执、embedding 缓存
  YYYY_MM_DD/progress.json     抽取阶段状态
  YYYY_MM_DD/publication.json  向量及发布阶段状态
upload_data/
  pic/YYYYMMDD/               待上传图片
  research-report-images/
    meta.json                 保留 180 天的批次列表
    batches/<batchVersion>/
      manifest.json
      data_part0000.jsonl.gz
```

新记录是 `op=upsert` 的协议记录。外层 `id` 与 `data.id` 都使用 `ResearchReportImages:` 前缀，`data.embedding` 为 2048 维 L2 归一化向量。

第一次发布使用 `YYYYMMDD000000000-0000`，同日继续有新增时递增最后序号。只创建新批次，历史文件不覆盖。分片压缩后不超过 10 MiB，新增量不生成未压缩的 JSONL 预览。

只有向量、分片及图片准备完成后才写 Manifest，最后更新 Meta。失败保留缓存，不把不完整批次加入 Meta。完整格式见 [发布协议](docs/upload_protocol.md)。

## Embedding 输入约定

固定系统指令：`Represent the user's input.`。用户内容先放图片，再放一个文本：

```text
Title: {title.strip()}
Content: {image_content.strip()}
```

保持图片预处理 `min_pixels=4096`、`max_pixels=1048576`、patch size 16，避免二次缩放。模型采用 BF16、最后一个有效 token pooling，输出转为 float32 并归一化。默认输入上限为 8192 token，不静默截断。

原工程的真实一致性样本未打包。若需要额外的分数回归测试，可以用 `EMBEDDING_VALIDATION_SAMPLE=/path/to/sample.json` 指向外部样本；省略时仍检查向量维度、有限值、归一化和记录对应关系。

`run_upload_embedding.sh --rewrite-unpublished` 仅用于旧格式、各批次 ID 不重复且确认尚未发布的整库补向量。日常增量使用 `run_increment.sh`，不要用整库重写命令处理已经发布的历史。

## 测试

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s . -p 'test_*.py'
```

默认测试用临时目录和合成输入，不启动 GPU，也不需要业务数据。依赖历史实物报告的集成测试默认跳过；只有具备其指定参考报告时才设置 `ENGREP_TEST_REPORT_DIR`。

## 代码范围

- `incremental_reports.py`、`check_increment.sh`、`run_increment.sh`：增量发现和任务编排。
- `extract_visual_records_light.py`、`extract_visual_records.py`、`visual_recall/`：抽取、裁图、质量检查和模型调用。
- `generate_search_queries.py`、`search_query_prompt.txt`：检索问句生成。
- `embed_upload.py`：embedding、缓存和内部增量编码子进程。
- `prepare_upload.py`：新批次、gzip 分片、Manifest 和 Meta。
- `run_qwen38_server.sh`、`run_embedding_server.sh`：可选的独立服务启动脚本。

本备份没有包括已搁置的 OPRD 实验、GPU 占用工具和历史实验产物。工作项目未被改动；导出副本仅做了路径参数化和去除对随仓库数据样本的强制依赖。
