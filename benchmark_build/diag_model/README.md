# CIR-QDiag 模型嵌入诊断

## 推荐实验：官方 PinPoint + 500-group 反事实 response profile

这里不在“原版 PinPoint”和“生成的 500 组”中二选一：

- 官方全量 7,635 queries 负责最终可比的 `mAP@10`、`NegRecall@10`；
- `pinpoint_pilot_500_v7` 是其中 500 个 group（1,933 queries）的机制诊断子集，只有它有
  审核后的 negative text，只负责 response profile 与直接 separation。

默认协议是 `simple_no_ml`：不训练 LogisticRegression，不运行 OOF probe，也不使用 learned
reranker。正式 rerank 是 baseline/hard 互相配合：一个模型召回 native Top-50，另一个模型
只用自己的正常 positive-query cosine similarity 重排，不使用任何 negative text。

推荐一次运行 Full 主比较、LoRA 复现，以及 image encoder 冻结组的机制对照（共六个模型，
逐个运行，不会同时占用显存）：

```bash
cd /mnt/data/datasets/search_dataset/zxl/benchmark_build/diag_model

CUDA_VISIBLE_DEVICES=0 bash run_counterfactual_profile_analysis.sh
```

如果想先只完成最关键、耗时最少的确认性实验，可只跑 Full baseline 与 Full Hard-0.5；主脚本
默认就是这一对：

```bash
CUDA_VISIBLE_DEVICES=0 python counterfactual_profile_analysis.py
```

主比较确认后，再用上面的 bash wrapper 补 LoRA 复现和 frozen-image 机制对照；已经完成的
negative-query cache 会自动复用。

正式编码前可先做只读式身份检查（官方 query、checkpoint fingerprint、各模型 corpus/query/Top-50
cache 必须严格匹配）：

```bash
bash run_counterfactual_profile_analysis.sh --preflight-only
```

现有官方 corpus/query/Top-50 cache 会直接复用；只需为每个 checkpoint 补编码约 3,242 条
strict negative query。中断后重复同一命令会复用已完成的 negative-query cache。若 embedding
已经完成、只想重算统计：

```bash
CUDA_VISIBLE_DEVICES=0 bash run_counterfactual_profile_analysis.sh --analysis-only
```

只想比较 baseline/hard 的原始 Top-K 候选集合，不计算 response profile：

```bash
python counterfactual_profile_analysis.py --ranking-jaccard-only
```

只重跑两个模型互相 rerank（无需 negative-query cache）：

```bash
python counterfactual_profile_analysis.py --cross-model-rerank-only
```

主 response 不要求每条 query 拥有相同数量或完整六类 negative。实现规则是：同类别内先
平均，再对该 query 实际存在的类别等权平均；缺失类别保留 `NaN + mask`，不填 0。完整六维
样本只报告覆盖率；预先指定的 `object_add + object_substitute` 二维 cohort 用于固定维度复核。
候选 separation 统计分别输出 native Top-10/20/50；rerank 也分别测试 10/20/50。

主要输出位于：

```text
../cir_qdiag_openai/outputs/pinpoint_pilot_500_v7/counterfactual_candidate_profiles/
  COUNTERFACTUAL_PROFILE_ANALYSIS.md
  MODEL_DECISION_SUMMARY.csv
  cohort_representativeness.csv
  negative_text_coverage_summary.csv
  target_exposure_audit_all_models.csv
  native_ranking_jaccard_summary.csv
  native_topk_retrieval_summary.csv
  cross_model_rerank_results.csv
  cross_model_rerank_by_query.parquet
  native_ranking_jaccard_by_query.csv
  rerank_results_all_models.csv
  shared_pool_paired_differences.csv
  <model>/candidate_profiles_native.parquet
  <model>/profile_dimension_stats.csv
  <model>/rerank/rerank_metrics_all_cohorts.csv
  shared_pool_comparisons/
```

官方 positive、官方 negative、未标注 Top-50 competitor 三类标签分开。Top-K 外的官方
positive 只追加到 separation 统计，不进入 rerank 候选。模型差同时报告各自 native Top-50
以及 baseline/hard Top-50 union 的固定候选口径，并按 V7 group 做 paired bootstrap。

`native_ranking_jaccard_summary.csv` 在官方全量和三个 Pilot cohort 上报告 Top-1/5/10/20/50
候选集合的 Jaccard、`|交集|/K`、平均交集数量与完全相同率；同一表还报告 baseline/hard 的
Positive Recall、HitRate、NegRecall、binary nDCG 及 paired delta/CI。Jaccard 忽略集合内部
顺序；跨模型 rerank 相对 retriever native ranking 的 Jaccard 写入
`cross_model_rerank_results.csv`。

默认只让原版 PinPoint evaluator 验证 native 和预注册的 Top-50 主 rerank，其余 variant 仍按
完全相同公式输出未舍入指标。如需让原版 evaluator 跑每个 alpha/depth：

```bash
CUDA_VISIBLE_DEVICES=0 bash run_counterfactual_profile_analysis.sh \
  --official-eval-all
```

非主 variant 的 evaluator 输入 JSON 使用临时目录，评估后不会留下数百 MB 的重复结果。
shared-union 的 candidate-level parquet 默认也不保存；确实需要逐候选文件时添加
`--save-shared-profiles`。

全量 rerank 表中的 `official_full_hybrid_pilot_only_rerank` 只修改有 profile 的 Pilot query，
其余官方 query 保持 native 排序；它用于显示对 7,635-query 总均值的稀释影响，不能解释成
profile 方法覆盖了完整官方数据。主要 rerank 证据看 `pilot_strict_negative_eligible` 及其
group bootstrap CI。

注意：V7 negative text 的生成/审核看过 target positives，因此 response-profile separation
仍只能解释为 target-informed diagnostic。跨模型 rerank 不读取 V7 negative text，不受这项
具体泄漏影响；它只评估两个已有模型的正常 q+ score 能否在另一模型召回池中提供互补排序。

代码会额外输出 `target_exposure_audit_all_models.csv`，在同一 query 内比较生成阶段列入
validated set 的 positives 与其它 official positives。这能检查明显的 target-specific response，
但后者仍共享同一语义/group，不能替代真正 target-blind 的独立测试集。

## PinPoint 官方全量评测（mAP@10 / NegRecall@10）

这条流程与下面的 V7 delta 诊断不同。它读取官方
`pinpoint_licensed.parquet` 的全部 7,635 条 query，在完整候选库上检索 Top-50，然后直接
调用 PinPoint 仓库原版 `evaluate.py`。最终对比表只保留官方 `mAP@10`、
`NegRecall@10` 和实际匹配的 query 数。

评估 frozen CLIP-L baseline 与 hard-weight=0.5，并复用之前已经生成的 frozen corpus cache：

```bash
cd /mnt/data/datasets/search_dataset/zxl/benchmark_build/diag_model

CUDA_VISIBLE_DEVICES=0 python run_clip_pinpoint_official.py \
  --models CLIP-L-baseline,CLIP-L-hard-w0.5 \
  --frozen-corpus-cache ../cir_qdiag_openai/outputs/pinpoint_pilot_500_v7/clip_l_pinpoint_diagnostics/_shared_frozen_clip_l_corpus \
  --batch-size 64
```

结果位于：

```text
../cir_qdiag_openai/outputs/pinpoint_official_full/
  official_map10_negrecall10.csv      # 最终只看这个
  missing_corpus_signatures.txt
  run_metadata.json
  <model>/<model>.json                # PinPoint 标准 Top-50 结果
  <model>/official_metrics.csv        # 原版 evaluate.py 的完整输出
```

如果不传 `--frozen-corpus-cache`，脚本会在输出目录重新建立 corpus cache。Full/LoRA 模型
不能共享 candidate embedding；训练完成后可分别选择模型运行：

```bash
CUDA_VISIBLE_DEVICES=0 python run_clip_pinpoint_official.py \
  --models CLIP-L-full-baseline,CLIP-L-full-hard-w0.5,CLIP-L-lora-r16-baseline,CLIP-L-lora-r16-hard-w0.5 \
  --batch-size 64
```

脚本可恢复并复用 corpus/query/search 缓存。只有明确需要覆盖旧缓存时才添加
`--force-corpus`、`--force-queries` 或 `--force-search`。

Full hard-weight 网格训练完成后，评估新增的四个权重：

```bash
CUDA_VISIBLE_DEVICES=0 python run_clip_pinpoint_official.py \
  --models CLIP-L-full-hard-w0.1,CLIP-L-full-hard-w0.25,CLIP-L-full-hard-w0.75,CLIP-L-full-hard-w1.0 \
  --output-root ../cir_qdiag_openai/outputs/pinpoint_official_full_hard_grid \
  --batch-size 64
```

结合已有 `pinpoint_official_full_lora/CLIP-L-full-{baseline,hard-w0.5}` 后，即可比较
`0.0、0.1、0.25、0.5、0.75、1.0` 的完整 mAP@10/NegRecall@10 曲线。

## 当前推荐：CLIP-L baseline 与 hard-weight=0.5 的 PinPoint 指标诊断

这条流程不训练 Logistic Regression、线性 probe、TF-IDF 或任何其他机器学习模型。
它只做两件事：用 checkpoint 抽取 embedding，然后在 109k PinPoint 全库上做精确检索并
计算 Delta、胜率、Recall@K 和 NDCG@K。

```bash
cd /mnt/data/datasets/search_dataset/zxl/benchmark_build/diag_model

CUDA_VISIBLE_DEVICES=0 python run_clip_pinpoint_diagnostics.py
```

默认比较：

- `CLIP-L-baseline`：`cir_qdiag_train/outputs/clip_l/batch64/baseline/last.pt`
- `CLIP-L-hard-w0.5`：`cir_qdiag_train/outputs/clip_l/batch64/hard_w0_5/last.pt`
- 数据：`cir_qdiag_openai/outputs/pinpoint_pilot_500_v7/final.jsonl`
- 输出：数据目录下的 `clip_l_pinpoint_diagnostics/`

两个 checkpoint 的 CLIP-L image backbone 都被冻结，因此脚本只编码一次 109k candidate
image corpus，第二个模型复用 `_shared_frozen_clip_l_corpus/`。query embedding 和检索结果按
模型分别缓存；任务中断后直接重跑同一命令会继续或复用缓存。

先用小规模检查环境和显存：

```bash
CUDA_VISIBLE_DEVICES=0 python run_clip_pinpoint_diagnostics.py \
  --limit-groups 5 \
  --limit-index 5000 \
  --output-root /tmp/clip_l_pinpoint_smoke
```

正式结果重点看：

- `comparison.csv`：两个模型 `all` 与 `strict`（ADD+SUBSTITUTE）的紧凑对比；
- `positive_retrieval_summary.csv`：原始正查询的 Recall@K、Hit@K、NDCG@K；
- `negative_metrics_summary.csv`：按 all/strict/operation/3×3 cell 展开的完整汇总；
- `<model>/analysis/negative_pair_metrics.csv`：每条 negative 的 Delta、胜率、Recall/NDCG 变化；
- `<model>/analysis/positive_query_metrics.csv`：每条原始 query 的检索指标。

这里的 `Delta = sim(q+, positives) - sim(q-, positives)`，大于 0 表示原始正确指令对原
positive images 的平均分高于 negative 指令。`raw win rate` 是 `Delta > 0` 的比例。
`strict` 不包含 DELETE，因为 DELETE 是更弱的查询，不一定应当降低原 positive 的得分。
`negative residual Recall/NDCG` 仍以原 positive image set 为 relevance，越低表示 negative
编辑越能把检索结果从原目标移开。Recall@K 使用标准 set recall（Top-K 命中数除以正样本
总数）；Hit@K 则表示 Top-K 是否至少命中一个正样本。

如果显存不足，可把 `--batch-size 128` 改为 64 或 32。`--force-corpus`、
`--force-queries`、`--force-search` 分别强制重算对应缓存。

完成 CLIP-L last-4 部分解冻训练后，用下面的命令做同样的 PinPoint 对比：

```bash
CUDA_VISIBLE_DEVICES=0 python run_clip_pinpoint_diagnostics.py \
  --models CLIP-L-last4-baseline,CLIP-L-last4-hard-w0.5 \
  --output-root ../cir_qdiag_openai/outputs/pinpoint_pilot_500_v7/clip_l_last4_pinpoint_diagnostics \
  --batch-size 64
```

两个 last-4 模型的 image encoder 分别经过训练，candidate embedding 不再相同。因此脚本
会为两个模型分别编码完整 109k corpus，不会复用 `_shared_frozen_clip_l_corpus`。

Full 与 LoRA 四组训练完成后运行：

```bash
CUDA_VISIBLE_DEVICES=0 python run_clip_pinpoint_diagnostics.py \
  --models CLIP-L-full-baseline,CLIP-L-full-hard-w0.5,CLIP-L-lora-r16-baseline,CLIP-L-lora-r16-hard-w0.5 \
  --output-root ../cir_qdiag_openai/outputs/pinpoint_pilot_500_v7/clip_l_full_lora_pinpoint_diagnostics \
  --batch-size 64
```

四个模型的 image encoder 都可能不同，所以四份 corpus embedding 分开计算。诊断读取最终
的 model-only `checkpoint_epoch_010.pt`，不会把 Full `last.pt` 中的大型 optimizer state
载入内存。

下面的 `run_diagnostics.py -> probe_grid.py` 是旧 CIR-QDiag embedding/probe 分析，和上述
PinPoint CLIP-L 比较无关；当前推荐命令不会调用它。

本目录针对 CIR-QDiag 的 `final.jsonl` 计算：

```text
r  = f_query(reference, empty modification)
q+ = f_query(reference, positive_text)
q- = f_query(reference, negative_text)
t  = f_candidate(target)
a+ = q+ - r
a- = q- - r
G+ = sim(q+, t) - sim(r, t) = dot(a+, t)
G- = sim(q-, t) - sim(r, t) = dot(a-, t)
v  = q+ - q-
delta = sim(q+, t) - sim(q-, t) = dot(v, t) = G+ - G-
```

`r` 始终使用 query 侧的无修改协议，而不是统一把 reference 当成 candidate：

- BGE-VL-base：只保留 reference image branch；
- BGE-VL-MLLM-S1：保留 query role 和 task prompt，modification 为 `None`；
- Qwen3：保留 composed-query instruction，仅输入 reference image。

因此 `G+`、`G-` 表示修改文本在给定 reference 条件下对 target score 的边际贡献，
其中仍包含图文交互，不应解释为脱离 reference 的“纯文本因果贡献”。

当前支持：

- `BGE-VL-base`
- `BGE-VL-MLLM-S1`
- `Qwen3-VL-Embedding-2B`

LinCIR 暂不加载。两个 BGE 模型复用
`/mnt/data/datasets/search_dataset/zxl/CIR_PLAY/run_pinpoint_bge_lincir.py`
中的原生 query/candidate 预处理。Qwen3 使用本地模型
`/mnt/data/datasets/search_dataset/zxl/qw3/Qwen3-VL-Embedding-2B` 自带的
`Qwen3VLEmbedder`，采用最后一个非 padding token pooling 和 L2 normalization：

```text
query  = reference image + modification text
         instruction: Represent the composed image retrieval query.
target = target image
         instruction: Represent the image for retrieval.
```

## 一键运行三个模型

在能够看到 CUDA GPU、且已经安装 `torch`、`transformers`、`pandas`、`numpy`、
`scikit-learn` 和 `Pillow` 的环境中运行：

```bash
cd /mnt/data/datasets/search_dataset/zxl/benchmark_build/diag_model

python run_diagnostics.py \
  --input ../cir_qdiag/outputs/pilot_50_v5/final.jsonl \
  --device cuda:0
```

默认输出到：

```text
../cir_qdiag/outputs/pilot_50_v5/model_diagnostics/
  BGE-VL-base/
  BGE-VL-MLLM-S1/
```

三个模型顺序运行，不会同时占用显存。Qwen3 默认 batch size 为 4；如果显存不足：

```bash
python run_diagnostics.py \
  --input ../cir_qdiag/outputs/pilot_50_v5/final.jsonl \
  --device cuda:0 \
  --qwen-batch-size 1
```

脚本默认 resume。每个 query/target embedding 都有独立 NPY 缓存，中断后重新执行同一命令即可。
`--force` 会重算当前输入所需的全部 embedding。

旧版本已经跑完 `q+`、`q-`、`t` 时，直接用同一命令 resume 即可：脚本只会新增每个唯一
reference 一次 `f_query(R, empty)`，随后重新组装特征；不会重算已有 query/target 缓存。

这次新增 `f(R)` 不会改变 `v=q+-q-`，因此无需重训原有线性 probe 和因子回归。只补
reference ablation 时使用快速模式：

```bash
python run_diagnostics.py \
  --input ../cir_qdiag/outputs/pilot_500_v5/final.jsonl \
  --device cuda:0 \
  --reference-ablation-only
```

该模式只刷新 `delta_by_cell.csv`、`reference_ablation_by_cell.csv` 和
`descriptive_report.json`，保留原有 probe/factor 结果。

只运行一个模型：

```bash
python run_diagnostics.py \
  --input ../cir_qdiag/outputs/pilot_50_v5/final.jsonl \
  --models BGE-VL-base \
  --device cuda:0
```

只把 Qwen3 补到已经跑完 BGE 的 `pilot_500_v5`：

```bash
python run_diagnostics.py \
  --input ../cir_qdiag/outputs/pilot_500_v5/final.jsonl \
  --models Qwen3-VL-Embedding-2B \
  --device cuda:0 \
  --qwen-batch-size 4
```

## 分阶段运行

只抽取 embedding：

```bash
python extract_embeddings.py \
  --input ../cir_qdiag/outputs/pilot_50_v5/final.jsonl \
  --model-name BGE-VL-base \
  --output-dir ../cir_qdiag/outputs/pilot_50_v5/model_diagnostics/BGE-VL-base \
  --device cuda:0 \
  --batch-size 64
```

只运行离线分析，不需要 GPU：

```bash
python probe_grid.py \
  --features-dir ../cir_qdiag/outputs/pilot_50_v5/model_diagnostics/BGE-VL-base \
  --output-dir ../cir_qdiag/outputs/pilot_50_v5/model_diagnostics/BGE-VL-base/analysis
```

## 抽取产物

```text
<model>/
  cache/query/*.npy       # 可恢复的 q+、q- 缓存
  cache/reference/*.npy   # 每个唯一 reference 的 query-side f(R) 缓存
  cache/target/*.npy      # 可恢复的 t 缓存
  features.npz            # q_reference、q_positive、q_negative、target、sample_index
  rows.jsonl              # negative 标签与 pair_id
  scores.csv              # 三个 sim、G+、G-、delta 及位移指标
  metadata.json
```

脚本会强制检查所有 embedding 的有限性和 L2 normalization，并验证：

```text
max(abs(delta - dot(v, t))) <= 1e-5
max(abs(delta - (G+ - G-))) <= 1e-5
```

## 分析产物

```text
<model>/analysis/
  delta_by_cell.csv       # 3×3，分别含 text/vision/all，pair bootstrap CI
  reference_ablation_by_cell.csv
                          # 3×3 的 sim(R,t)、G+、G-、位移和局部方向统计
  probe_metrics.csv       # 所有线性 probe 的 OOF 指标
  confusion_*.csv         # confusion matrices
  factorial_report.json   # additive 与完整 9-cell 模型比较
  oof_predictions.jsonl   # 每条 negative 的 out-of-fold 预测
  report.json
  summary.md
  vector_visualization/     # 可选的 PCA/t-SNE 与无监督聚类诊断
```

主要 probe：

- `raw_v -> 9 cells`
- `raw_v -> semantic_target`
- `raw_v -> edit_operation`
- `direction_v -> 9 cells`
- `norm_v`、`q_positive`、`q_negative` 控制组
- `(positive_text, negative_text)` TF-IDF 文本泄漏基线
- text-only 与 vision-grounded 之间的跨模式泛化

所有交叉验证都按 `pair_id` 分组，同一 CIRR sample 的 negatives 不会同时出现在训练和测试中。
正则强度只在外层训练 fold 内选择。准确率、macro-F1 和 balanced accuracy 使用 pair-level
bootstrap 置信区间。

## 编辑向量降维与聚类可视化

已经生成 `features.npz` 后，不需要 GPU，也不需要重新训练线性 probe：

```bash
python run_diagnostics.py \
  --input ../cir_qdiag/outputs/pilot_500_v5/final.jsonl \
  --models BGE-VL-base,BGE-VL-MLLM-S1 \
  --output-root ../cir_qdiag/outputs/pilot_500_v5/model_diagnostics \
  --visualization-only
```

也可以直接运行单模型：

```bash
python visualize_vectors.py \
  --features-dir ../cir_qdiag/outputs/pilot_500_v5/model_diagnostics/BGE-VL-base \
  --output-dir ../cir_qdiag/outputs/pilot_500_v5/model_diagnostics/BGE-VL-base/analysis/vector_visualization
```

默认分别分析：

- `raw_v = q+ - q-`：同时包含位移方向与长度；
- `direction_v = v / ||v||`：只看方向是否形成 3×3 结构。

每个视图先降到 PCA-50，再生成 PCA-2D 和 t-SNE-2D。图中分别按 9-cell、
semantic target、edit operation 和 construction mode 着色。输出包括：

```text
vector_visualization/
  pca_raw_v.png
  tsne_raw_v.png
  pca_direction_v.png
  tsne_direction_v.png
  coordinates.csv
  quality_metrics.csv
  report.json
  summary.md
```

`report.json` 同时给出 PCA 分析空间中的 silhouette、邻域标签纯度、KMeans
ARI/NMI，以及二维投影的 trustworthiness。t-SNE 只用于观察局部结构；如果二维图出现
“岛屿”，但原 PCA 分析空间的 silhouette、邻域纯度和 KMeans 指标都很低，不应把它解释为
真实的 3×3 聚类。

### PinPoint V7

PinPoint V7 使用自己的 query cache，不能读取旧 CIR-QDiag 的 `features.npz`。已经完成
full-corpus diagnostics 后可直接离线运行：

```bash
python run_pinpoint_full_diagnostics.py \
  --input ../cir_qdiag_openai/outputs/pinpoint_pilot_500_v7/final.jsonl \
  --models BGE-VL-base,BGE-VL-MLLM-S1 \
  --output-root ../cir_qdiag_openai/outputs/pinpoint_pilot_500_v7/full_retrieval_diagnostics \
  --visualization-only
```

该命令从每个模型已有的 7,509 条 query embedding cache 中配对原 instruction 与 accepted
negative，计算 5,576 个 `v=q+−q-`；不会重算 query、109,598 张 corpus embedding 或全库
Top-K。第四个着色面板使用 `fully_reviewed/partially_reviewed`，用于检查“至多审核 4 张
Positive”是否成为几何混杂因素。

PinPoint V7 的 Delta 分布图使用专用脚本：

```bash
python analyze_pinpoint_delta_modes.py \
  --input ../cir_qdiag_openai/outputs/pinpoint_pilot_500_v7/final.jsonl \
  --diagnostics-root ../cir_qdiag_openai/outputs/pinpoint_pilot_500_v7/full_retrieval_diagnostics \
  --bootstrap-replicates 5000
```

输出仍沿用目录名 `delta_mode_analysis/`，但 V7 不存在旧数据中的
`text_only_editable/vision_grounded` construction-mode 对。V7 图表按真实存在的
ADD/DELETE/SUBSTITUTE、3×3 cell 和 `fully_reviewed/partially_reviewed` 展开，并同时
生成理论 `[-2,2]` 全范围直方图与经验区间放大图。

`Delete` negative 是更弱查询，不要求和 target 冲突。因此 `delta <= 0` 对 Delete 只作描述性
统计，不应直接当作检索错误。

`delta_by_cell.csv` 同时给出每格的 `sim(q-,t)`、`sim(q+,t)`、`q-→q+` 球面角度和
局部 target-gradient 对齐。`reference_ablation_by_cell.csv` 给出每格的：

- `sim(r,t)`、`sim(q+,t)`、`sim(q-,t)`；
- `G+` 与 `G-` 的均值、中位数、正贡献率和 pair-bootstrap CI；
- `r→q+`、`r→q-` 的 chord 距离、球面角度和局部梯度对齐；
- 两个 modifier 位移 `a+`、`a-` 的方向余弦。

该文件同时保留 negative-row mean，并给出 `pair_balanced_mean_*` 作为主比较口径：先在
每个 pair×cell 内求均值，再让每个 pair 等权聚合；相应 CI 也对这些 pair-level 均值重采样。
这可避免同一 pair 在 `all` 模式下同时具有 text/vision negative 时被重复加权。

九格标签描述的是 `m+ → m-` 的扰动类型。同一 sample 的 `r` 和 `q+` 可能出现在多个
negative cell 中，因此单格的 `sim(r,t)` 与 `G+` 是该格所覆盖样本的条件均值，不应解释为
该 cell 独立地产生了 `m+`。

## 500 条正式诊断

pilot-50 适合检查工程是否正确。生成 pilot-500 后直接替换输入：

```bash
python run_diagnostics.py \
  --input ../cir_qdiag/outputs/pilot_500_v5/final.jsonl \
  --device cuda:0
```

## Delta construction-mode 比较与分布图

三个模型完成后可离线运行，不需要 GPU：

```bash
python analyze_delta_modes.py \
  --diagnostics-root ../cir_qdiag/outputs/pilot_500_v5/model_diagnostics \
  --bootstrap-replicates 5000
```

### Positive-paraphrase Delta calibration

When a `positive_paraphrases.jsonl` file is available, calibrate how much
`sim(f(R,m+),t)` changes under accepted meaning-preserving rewrites:

```bash
python run_paraphrase_robustness.py \
  --input ../cir_qdiag_openai/outputs/pilot_50_v6_openai_terra/final.jsonl \
  --paraphrases ../cir_qdiag_openai/outputs/pilot_50_v6_openai_terra/positive_paraphrases.jsonl \
  --diagnostics-root ../cir_qdiag_openai/outputs/pilot_50_v6_openai_terra/model_diagnostics \
  --models BGE-VL-base,BGE-VL-MLLM-S1,Qwen3-VL-Embedding-2B \
  --device cuda:0
```

For each model the script estimates the pair-balanced empirical interval of

```text
epsilon = paraphrase_positive_score - original_positive_score
```

and maps each original Delta to `[Delta + L, Delta + U]`. A stable win requires
the lower endpoint to remain above zero; a stable loss requires the upper
endpoint to remain below zero; all other cases are uncertain. Calibration uses
leave-one-pair-out intervals for scoring and a pair bootstrap for endpoint
uncertainty. Existing target/negative embeddings are reused.

Outputs are written under each model's `paraphrase_robustness/` directory:

- `paraphrase_scores.csv`: per-paraphrase score shifts;
- `robust_negative_scores.csv`: calibrated and directly observed Delta ranges;
- `summary.csv`: pair-balanced original win, observed-variant expected win,
  stable win, and uncertainty rates;
- `calibration.json`: empirical interval and bootstrap confidence intervals.

The diagnostics root also receives `paraphrase_robustness_summary.csv`.

The calibration run re-encodes the original positive text alongside its
paraphrases. Therefore `epsilon` compares queries produced by the same runtime
device/dtype, while `original_rescore_drift` records any difference from the
previous diagnostic run. Calibration cache keys include the model path,
device, and dtype, so a CPU verification run cannot be reused by a later GPU
run.

## PinPoint V7 full-corpus diagnostics

PinPoint V7 has a group-level schema rather than the single-target CIRR schema:
each group contains 2--6 native positive-instruction paraphrases, 1--2
reference images, a shared positive-image set, and accepted generated negative
text. Run the dedicated full-corpus pipeline for Base and MLLM-S1:

```bash
python run_pinpoint_full_diagnostics.py \
  --input ../cir_qdiag_openai/outputs/pinpoint_pilot_500_v7/final.jsonl \
  --models BGE-VL-base,BGE-VL-MLLM-S1 \
  --device cuda:0 \
  --ks 1,5,10,50 \
  --bootstrap-replicates 1000
```

For each model, the pipeline caches the full PinPoint candidate-image
embeddings, all original/negative composed-query embeddings, and exact
full-corpus Top-50 results. An interrupted corpus or query extraction resumes
from its progress file.

The native positive paraphrases estimate a model-specific normal-perturbation
threshold:

```text
tau = group-balanced Q95(
    abs(sim(f(R,m_i+),t) - sim(f(R,m_j+),t))
)
```

A generated negative wins only when the source-positive margin exceeds `tau`.
NDCG uses the original positive image set as binary relevance over the complete
PinPoint corpus. Thus negative-text NDCG is residual positive retrieval, not
accuracy for the negative instruction. Top-K Jaccard is computed both between
native positive paraphrases and between each source instruction and its
generated negative.

Outputs under each model's `analysis/` directory:

- `positive_query_metrics.csv`: full-corpus NDCG per native positive query;
- `negative_pair_metrics.csv`: target win rates, positive/negative NDCG, NDCG
  gap, and source/negative Top-K Jaccard;
- `positive_paraphrase_pair_metrics.csv`: retrieval-set Jaccard and NDCG
  variation among native positive paraphrases;
- `negative_summary.csv`: group-balanced all/strict/operation/3x3 summaries;
- `calibration.json`: normal-perturbation threshold and group-bootstrap CI;
- `positive_summary.json`: positive NDCG and paraphrase retrieval consistency.

`DELETE` is reported separately because PinPoint V7 deliberately accepts it as
a weaker query without requiring contradiction of the positive images.

`pair_balanced_observed_variant_win_rate` first averages the win indicator over
the original wording plus that sample's accepted paraphrases, then gives every
pair equal weight. It is the appropriate average-case paraphrase metric when
different samples have different numbers of accepted rewrites.

输出到 `model_diagnostics/delta_mode_analysis/`：

- `mode_win_rates_by_cell.csv`：model × mode × 3×3 cell 的 Delta 均值、胜率、
  pair-bootstrap CI 和 Wilson CI；
- `mode_comparisons.csv`：pair-balanced、cell-macro 和 matched pair×cell 三种
  text/vision 比较；所有差值均定义为 `vision - text`；
- `delta_hist_by_model_mode_full_range.png`：理论范围 `[-2,2]`、纵轴样本数；
- `delta_hist_by_model_mode_zoom.png`：跨模型共同经验区间放大；
- `delta_hist_3x3_<model>.png`：每个模型完整九格的 text/vision count histogram；
- `delta_win_rate_3x3_by_mode.png`：九格胜率及95% Wilson CI；
- `delta_win_rate_mode_summary.png`：总体与严格集合的 mode 胜率摘要。

`cell-macro` 让每个 cell 等权，用于控制 text/vision 的九格组成差异；
`matched_pair_cell` 只比较同一 source pair、同一 cell 同时存在两种 construction mode
的情况。后者仍不是随机化因果实验，因为两种 negative text 并不相同。
