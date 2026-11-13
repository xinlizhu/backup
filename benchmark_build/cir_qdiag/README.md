# CIR-QDiag V5 for CIRR

CIR-QDiag 根据 CIRR validation 三元组
`(reference image, positive modification text, target image)` 自动生成 query-side hard-negative
text，用于检验：

```text
score(reference, positive, target) > score(reference, negative, target)
```

V5 的重点是简单：不解析 claims，不先规划 mutation，也不串联多层逐候选判别。每条原始样本
先调用 VLM 1 次做保守的 CIRR 源数据筛查；通过后调用 6 次生成和 2 次批量审核，共 9 次。

## 分类与两种生成方式

每个候选由三个轴描述：

```text
semantic_target    = object | attribute | predicate
edit_operation     = add | delete | substitute
construction_mode  = text_only_editable | vision_grounded
```

- `object`：实体存在、身份或类别。
- `attribute`：颜色、材质、大小、数量、状态、视角或构图等一元属性。
- `predicate`：动作、姿态或实体间关系，包括空间关系。

Text-only 只读取 positive text。Vision-grounded 读取 positive text 和唯一一张 Target 图像，
根据 Target 中可见的对象、属性和关系构造能被 Target 直接反驳的近邻反事实；生成模型不读取
Reference 图像。

每种模式分别运行 Add、Delete、Substitute 三次；六个调用使用六份独立 prompt。每次调用只看
当前一种操作，并同时返回 object、attribute、predicate 三个完整 negative text 或 `null`。
因此每条通过筛查的样本最多生成 18 条，但模型无需凑满。

## 实际流程

```text
CIRR sample
  -> 1 次双图源样本筛查（keep / drop）
  -> drop 写入 source_rejections.jsonl 并停止处理
  -> 单条请求异常写入 source_failures.jsonl，其余样本继续
  -> 3 次 text-only 直接生成
  -> 3 次 vision-grounded 直接生成
  -> 本地校验编辑锚点，并去除空值、原文和重复文本
  -> 1 次 Target-only 批量筛选 Add/Substitute
  -> 1 次 Target-only 独立批量筛选 Delete
  -> final.jsonl
```

源样本筛查只淘汰文本不可理解、核心 Target 命题明确错误、显式相对关系明显矛盾或完全无法形成
组合检索解释的三元组；不确定时保留，也不会因为两张图不是字面编辑关系而淘汰。

两个候选筛选器分别把各自候选分入 `selected` 或 `rejected`，每条只附一个简短原因。Delete 独立
审核是因为它允许与 Target 相容，而 Add/Substitute 必须与 Target 冲突。系统没有
positive-transition 全局审核，也没有逐候选 text/visual/final 三重审核。

CIRR 是组合图像检索，不是字面意义的图像编辑。Reference 和 Target 可以是不同照片、不同主体
或不同场景；positive text 也不需要穷举全部视觉差异。源样本筛查仍读取两张图，以便检查
`more`、`another`、`farther`、`turn X into Y` 等显式相对表达；负样本生成和最终候选审核只读取
Target。

筛选规则：

- 所有候选必须自然、接近 positive、只做一个主要语义编辑，并且标签正确。
- 每条候选显式保存 `before_span -> after_span`：Add 为 `null -> 新增短语`，Delete 为
  `被删短语 -> null`，Substitute 为 `原短语 -> 替换短语`。旧值必须真实出现在 positive，新值
  必须真实出现在 negative，不能从 Target 图像反推一个文本中不存在的旧值。
- positive text 决定 Add/Delete/Substitute 以及 Object/Attribute/Predicate；Target 图像只证明
  Add/Substitute 的新事实是否为假。属性只能依附 positive 已命名实体，谓词不能暗中新增参与者。
- Add/Substitute 的新命题应当与 Target 清楚不兼容。
- Delete 必须删掉一个真实约束，剩余句子自然且严格更弱；不要求与 Target 矛盾。
- Vision-grounded 的 `visual_basis` 必须指出 Target 中直接反驳候选事实的可见证据；Text-only
  不保存视觉依据。

系统没有人工审批步骤。`audit.html` 只是展示自动结果。

## 安装与 vLLM

```bash
cd /mnt/data/datasets/search_dataset/zxl/benchmark_build/cir_qdiag
python -m pip install -e .
python -m pip install -e '.[dev,evaluation]'
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

`max_num_batched_tokens` 同时影响多模态 encoder cache 的预分配预算；只增加
`max_model_len` 不能解决 `pre-allocated encoder cache size` 报错。

## 运行

```bash
curl http://127.0.0.1:8000/v1/models
python scripts/smoke_test_vllm.py --config configs/pilot_20.yaml
python scripts/run_pipeline.py --config configs/pilot_20.yaml --resume
```

完整流水线运行时会分别显示三个按样本计数的进度条：

```text
Screen CIRR
Generate negatives
Validate negatives
```

进度条包含已完成样本数、处理速度和预计剩余时间。Generate 与 Validate 支持断点续跑，并从
已有输出中的完成样本数开始显示。进度在一条样本完成该阶段全部模型调用并写入 JSONL 后更新。

### 只生成语义等价的正确文本

下面的独立命令不会运行 CIRR 筛查、九宫格 hard-negative 生成或 negative 审核。它对输入
JSONL 中每条 `positive_text` 只执行两次纯文本 Qwen 调用：

```text
positive_text
  -> 批量生成 5 条语义等价改写
  -> 批量审核与原文是否完整等价
  -> 保留审核通过的正确改写并计算文本扰动
```

推荐读取已经通过源样本筛查的 `screened.jsonl`：

```bash
python scripts/generate_positive_paraphrases.py \
  --config configs/pilot_50.yaml \
  --input outputs/pilot_50_v5/screened.jsonl \
  --output outputs/pilot_50_v5/positive_paraphrases.jsonl \
  --num-variants 5 \
  --force
```

运行时会显示按样本计数的进度条、处理速度和预计剩余时间。断点续跑时，进度条从输出中已经完成的
样本数开始；`accepted` 和 `generated` 显示最近完成样本的审核通过数与候选数。

生成和审核均不读取 Reference 或 Target 图片，也不会利用图片补充文本中不存在的信息。审核只检查
对象、数量、属性、动作、关系、方向、比较和指代是否与原始 positive text 完整等价。输出包括：

```text
positive_paraphrases.jsonl          # 原文、全部候选、通过的等价改写和调用记录
positive_paraphrase_metrics.csv     # 每条通过改写的文本扰动指标
positive_paraphrase_summary.json    # 接受率及扰动分布汇总
cache/generate_positive_paraphrases/
cache/review_positive_paraphrases/
```

扰动指标包括词数比、字符编辑距离、归一化字符编辑距离、词级编辑距离、归一化词级编辑距离和
token Jaccard。默认参数位于配置的 `paraphrase` 段；命令行 `--num-variants` 可覆盖生成条数。

V5 输出位于 `outputs/pilot_20_v5`：

```text
source.jsonl
screened.jsonl
source_rejections.jsonl
source_failures.jsonl
candidates.jsonl
final.jsonl
audit.html
generation_by_edit.csv
selection_by_edit.csv
selection_reasons.csv
sample_summary.csv
final_distribution.csv
cache/<stage>/<input_hash>.json
```

`source_rejections.jsonl` 只保存模型明确判定无效的 CIRR 样本；`source_failures.jsonl` 保存超出
上下文、服务暂时错误等操作性失败。单条失败不会终止整批，重新运行 pipeline 时会再次筛查这些
source 样本。

V5 schema 与 V4 不兼容，配置使用新目录和 `prompt_version: v5`，不会混入旧结果或缓存。逐阶段
命令见 [run.md](../run.md)。字段说明见
[HARD_NEGATIVE_AUXILIARY_ARTIFACTS.md](HARD_NEGATIVE_AUXILIARY_ARTIFACTS.md)。

## 验收

```bash
PYTHONPATH=src python scripts/export_schemas.py
ruff check src scripts tests
PYTHONPATH=src pytest -q
python -m compileall -q src scripts tests
```
