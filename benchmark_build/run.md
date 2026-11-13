占卡
bash /mnt/data/datasets/search_dataset/zxl/engReport_code/run_qwen38.sh
bash /mnt/data/datasets/search_dataset/zxl/engReport_code/run_qwen38.sh stop
bash /mnt/data/datasets/search_dataset/zxl/engReport_code/run_qwen38.sh status


看内存
bash /mnt/data/datasets/search_dataset/zxl/check_system_disk.sh

环境
source /mnt/data/datasets/search_dataset/zxl/for_conda/bin/activate
conda activate /mnt/data/datasets/search_dataset/zxl/multi_modal_py3.12


增量（抽取 + embedding + 本地发布包）：
cd /mnt/data/datasets/search_dataset/zxl/engReport_code
bash check_increment.sh --start-date 2026-06-01
bash run_increment.sh --start-date 2026-06-01

# check 只读统计新增和失败重试；run 不上传 OSS。
# run：抽取服务关闭后，自动计算待发布记录的 Qwen3-VL-Embedding-2B 向量，再生成新 batch。
# 抽取默认端口 7888、显存预算 0.85；embedding 默认显存预算 0.95。
# embedding 可单独设置：--embedding-gpu-memory-utilization 0.95 --embedding-batch-size 256
# --batch-date 可选，默认北京时间当天。新增批次不覆盖历史批次；同日新增递增末尾序号。
# upload_data/pic/YYYYMMDD/：图片；upload_data/research-report-images/：Meta、Manifest 和 gzip 分片。
# 新记录包含 data.embedding（2048 维、L2 归一化），不再生成 data_part*.jsonl 预览。
# data/ 保留本地基础索引和新增图片；日志、问句、向量缓存和发布状态放 data_mid/。
# 发布状态：data_mid/YYYY_MM_DD/publication.json；抽取状态：同目录 progress.json。
# embedding 失败时不更新 Meta，重跑会复用已完成抽取及向量缓存。

# 仅补算/发布已经抽取的数据（不启动 27B 抽取模型；日期改为实际发布日）：
python prepare_upload.py --batch-date 20261106

上传到 OSS：
# 本地待上传目录为 engReport_code/upload_data，不能再直接同步整个 data/。
# 先上传 pic/ 到 Bucket 的 pic/；再将 research-report-images/ 中的 gzip 分片上传到
# prod/research-report-images/ 对应批次；校验成功后上传 Manifest，最后合并远端 Meta。
# 不上传用于本地查看的未压缩 JSONL；不要直接 sync 整个目录，以免 Meta 提前可见。


MAX_NUM_BATCHED_TOKENS=16000 PORT=8088 bash /mnt/data/datasets/search_dataset/zxl/engReport_code/run_embedding_server.sh

PORT=8088 bash /mnt/data/datasets/search_dataset/zxl/engReport_code/run_qwen38_server.sh

cd /mnt/data/datasets/search_dataset/zxl/vla_repro/envs
conda activate ./openvla-oft

export OPENAI_BASE_URL="https://ai.gs88.shop/v1"
export OPENAI_API_KEY="[REDACTED]"
python scripts/run_pipeline.py --config configs/pilot_50.yaml

# CIR-QDiag V5 运行命令
export OPENAI_API_KEY="[REDACTED]"
## 1. 环境与 vLLM
cd /mnt/data/datasets/search_dataset/zxl/benchmark_build/diag_model

python run_diagnostics.py \
  --input /mnt/data/datasets/search_dataset/zxl/benchmark_build/cir_qdiag_openai/outputs/pilot_50_v6_openai_terra \
  --models BGE-VL-MLLM-S1,Qwen3-VL-Embedding-2B,BGE-VL-base \
  --device cuda:0 \
```bash
cd /mnt/data/datasets/search_dataset/zxl/benchmark_build/cir_qdiag
source /mnt/data/datasets/search_dataset/zxl/for_conda/bin/activate
conda activate /mnt/data/datasets/search_dataset/zxl/multi_modal_py3.12
python -m pip install -e .
```

单张 H100：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/serve_vllm.py \
  --config configs/model.yaml \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --max-num-batched-tokens 32768 \
  --max-num-seqs 2
```

四张 H100：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/serve_vllm.py \
  --config configs/model.yaml \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --max-num-batched-tokens 32768 \
  --max-num-seqs 8
```

多模态报错中的 `pre-allocated encoder cache size` 由 `max_num_batched_tokens` 约束，不等同于
`max_model_len`。默认值也写在 `configs/model.yaml`。

```bash
curl http://127.0.0.1:8000/v1/models
python scripts/smoke_test_vllm.py --config configs/pilot_20.yaml
```

## 2. 一键运行 20 条

每条原始样本先执行 1 次 CIRR 源数据筛查。保留的样本再执行 6 次直接生成和 2 次按操作语义
分开的批量筛选。每次生成只处理一种操作，并同时生成 object、attribute、predicate 三项。

```bash
python scripts/run_pipeline.py --config configs/pilot_20.yaml --resume
```

输出目录是 `outputs/pilot_20_v5`。V4 结果不会被覆盖或复用。

## 3. 逐阶段运行

```bash
python scripts/prepare_cirr.py \
  --config configs/pilot_20.yaml \
  --output outputs/pilot_20_v5/source.jsonl

python scripts/screen_cirr.py \
  --config configs/pilot_20.yaml \
  --input outputs/pilot_20_v5/source.jsonl \
  --output outputs/pilot_20_v5/screened.jsonl \
  --rejected-output outputs/pilot_20_v5/source_rejections.jsonl

python scripts/generate_negatives.py \
  --config configs/pilot_20.yaml \
  --input outputs/pilot_20_v5/screened.jsonl \
  --output outputs/pilot_20_v5/candidates.jsonl

python scripts/validate_negatives.py \
  --config configs/pilot_20.yaml \
  --input outputs/pilot_20_v5/candidates.jsonl \
  --output outputs/pilot_20_v5/final.jsonl
```

重新生成候选或只重跑最终筛选：

```bash
python scripts/run_pipeline.py --config configs/pilot_20.yaml --resume --force-stage generate
python scripts/run_pipeline.py --config configs/pilot_20.yaml --resume --force-stage validate
```

## 4. 报告与 500 条 pilot

```bash
python scripts/export_audit.py \
  --input outputs/pilot_20_v5/final.jsonl \
  --output outputs/pilot_20_v5/audit.html

cir-qdiag report \
  --input outputs/pilot_20_v5/final.jsonl \
  --output-dir outputs/pilot_20_v5

python scripts/run_pipeline.py --config configs/pilot_500.yaml --resume
```

报告和 `audit.html` 都是自动结果展示，不是人工审批环节。

## 5. 离线验收

```bash
PYTHONPATH=src python scripts/export_schemas.py
ruff check src scripts tests
PYTHONPATH=src pytest -q
python -m compileall -q src scripts tests
```
