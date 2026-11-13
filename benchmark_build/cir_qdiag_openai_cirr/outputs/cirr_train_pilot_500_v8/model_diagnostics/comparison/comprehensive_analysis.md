# CIR-QDiag train-500 全面诊断报告

生成日期：2026-08-03

## 评测范围

- 500 个 CIRR train pair，491 张唯一 reference、493 张唯一 target。
- 3,507 条生成候选，2,446 条经 Reference + Target 视觉审核后接受。
- 严格 hard negative 为 Add + Substitute：1,695 条，覆盖 493/500 pair。
- Delete 共 751 条，是删除信息后的弱查询，单独作描述性分析。
- 7 个模型的 embedding、paraphrase calibration、5-fold grouped probe 均已完成。
- 所有 probe 按 `pair_id` 分组划分，避免同一个 pair 跨 train/test。

## 一、数据集构建分析

### 1. 构建完整性与审核

- 500/500 样本 `operational_status=ok`，视觉验证模式均为
  `reference_target_visual`。
- 2,446/3,507 候选通过，整体接受率 69.75%。
- 每个样本平均接受 4.892 条、median=5；最少 1 条、最多 9 条。
- 31 个样本的全部生成候选通过；469 个样本同时含通过与拒绝候选，因此
  `semantic_status=partial`。这里的 partial 不表示运行失败。
- 接受样本中没有同一 pair 内的重复 negative 文本，也没有重复 3×3 cell；所有
  accepted negative 都有视觉审核理由。
- 1,061 条拒绝中：747 条 `not_counterfactual`、169 条
  `other_semantic_failure`、85 条 `invalid_delete`、31 条
  `wrong_semantic_target`、24 条 `not_fluent`、5 条 `not_single_edit`。

### 2. 3×3 分布与选择偏差

| Cell | 通过/生成 | 接受率 | 7 模型平均 Delta 胜率 |
|---|---:|---:|---:|
| object × add | 360/464 | 77.59% | 80.83% |
| object × delete | 141/180 | 78.33% | 73.15% |
| object × substitute | 390/468 | 83.33% | 89.71% |
| attribute × add | 198/496 | 39.92% | 78.43% |
| attribute × delete | 338/367 | 92.10% | 64.07% |
| attribute × substitute | 301/357 | 84.31% | 78.69% |
| predicate × add | 161/461 | 34.92% | 65.48% |
| predicate × delete | 272/313 | 86.90% | 63.97% |
| predicate × substitute | 285/401 | 71.07% | 68.22% |

关键结论：

- 9 个 cell 都有数据，但最大 cell（object × substitute，390）是最小 cell
  （object × delete，141）的 2.77 倍。
- attribute/predicate Add 的审核通过率只有 39.92%/34.92%，说明这两类生成最困难，
  通过后的样本也可能偏向“视觉上更明显、更容易审核”的子集。
- object × substitute 对所有模型都最容易，平均胜率 89.71%；严格负样本中
  predicate × add/substitute 最困难，平均胜率 65.48%/68.22%。
- Delete 的语义目标与 Add/Substitute 不同：它只是移除约束，不保证与 target
  矛盾，不能与严格 hard negative 混合排名或混合训练权重。

### 3. 已发现的明确数据问题

有 2 条 accepted Delete 与 positive 文本完全相同，应删除或重新生成：

- `cirr_train_3805_object_delete`
- `cirr_train_25414_attribute_delete`

比例仅为 2/2,446（0.082%），不会改变总体结论，但它们属于明确 no-op，说明视觉审核
理由与最终文本之间还需要程序化一致性检查。

### 4. 数据多样性与潜在泄漏

- 所有 negative 都是 `text_only_editable`：生成阶段没有视觉输入，只有审核阶段看
  Reference + Target。因此该版本不覆盖 vision-grounded negative construction。
- 同一模型同时担任 generator 和 visual validator，可能带来相关性偏差；后续可用独立
  validator 或人工抽检估计这种偏差。
- TF-IDF 9-cell probe 准确率为 27.15%，高于均匀机会 11.11%，说明文本表面仍有一定
  cell 信号，但远低于 edit-vector direction probe 的 54.70%–64.47%。
- `q_positive` 单独预测 cell 只有 11.24%–12.47%，接近机会水平；`q_negative`
  单独为 29.11%–33.40%，而差向量 direction 达 54.70%–64.47%。这说明主要标签信号
  确实来自正负编辑差异，而不是 pair 身份泄漏。
- 9-cell centroid 的 additive held-out R² 仅 1.29%–3.71%，saturated R² 仅
  2.03%–5.18%。数据并未坍缩为少量固定模板，cell 内仍有很高的内容多样性。

### 5. Paraphrase 校准数据

- 500 个样本均有 operational-ok 记录；1,390 条 paraphrase 通过审核，覆盖
  494/500 样本。
- Paraphrase 只用于估计正确 positive 的正常分数波动
  `epsilon = score(paraphrase) - score(original)`。
- 对每条原始 negative 使用 `[Delta + L, Delta + U]`；下界大于 0 才是
  `stable_win`。没有计算文本 Jaccard，也没有计算“paraphrase 胜率”。

## 二、整体模型分析

### 1. 综合指标

检索 Recall/NDCG 使用现有 493 张唯一 target 作为候选库，标记为
`train500_target_pool`，不是标准 CIRR group/full-gallery Recall。

| 模型 | 严格 Delta 胜率 | Mean Delta | 校准后 stable win | Target-pool R@1 | R@10 | MRR | Direction 9-cell probe |
|---|---:|---:|---:|---:|---:|---:|---:|
| BGE-VL-base | 72.78% | 0.0276 | 14.08% | 66.40% | 96.40% | 0.7836 | 62.14% |
| BGE-VL-large | 77.42% | 0.0329 | 17.68% | 71.20% | 97.20% | 0.8109 | 61.00% |
| BGE-VL-MLLM-S1 | 78.67% | 0.0271 | 31.72% | 74.20% | 97.80% | 0.8323 | **64.47%** |
| BGE-VL-MLLM-S2 | 81.72% | 0.0292 | **36.99%** | **80.60%** | **99.00%** | **0.8778** | 63.16% |
| Qwen3-VL-Embedding-2B | **83.73%** | **0.0416** | 22.99% | 72.40% | 98.20% | 0.8175 | 54.70% |
| LinCIR | 80.53% | 0.0178 | 26.52% | 52.20% | 93.40% | 0.6715 | 60.92% |
| Pic2Word | 78.42% | 0.0170 | 18.16% | 53.00% | 91.80% | 0.6627 | 59.53% |

### 2. 模型结论

#### BGE-VL-MLLM-S2

- 综合表现最好：target-pool R@1、R@10、MRR 和 paraphrase 校准后的 stable-win
  比例均为第一。
- 严格 Delta 胜率 81.72%，仅次于 Qwen。
- 相比 S1，严格胜率提升 3.05pp、R@1 提升 6.4pp、stable win 提升 5.27pp。
- 是当前最适合作为主训练/评测参考的单模型。

#### Qwen3-VL-Embedding-2B

- 原始严格 Delta 胜率 83.73%、Mean Delta 0.0416，均为第一。
- 但 positive paraphrase 的正常波动区间较宽，stable win 只有 22.99%；高原始 margin
  不等于高措辞稳定性。
- Target-pool R@1 72.4%，落后 S2/S1；正负 Top-10 Jaccard 0.7189，为所有模型最高，
  说明单次错误编辑通常不会大幅改变其 Top-10 检索集合。
- 9-cell direction probe 最低，说明它的编辑几何更难被简单线性分类，不等同于性能差。

#### BGE-VL-MLLM-S1

- 整体稳定的第二梯队；stable win 31.72%，仅次于 S2。
- Direction 9-cell probe 64.47% 为最高，说明它的 edit vector 对 3×3 操作最线性可分。
- S2 在检索和 Delta 上均进一步提升，但 S1 的结构诊断信号略更清晰。

#### BGE-VL-base / large

- large 相比 base：严格胜率 +4.64pp、R@1 +4.8pp、Mean Delta +0.0053。
- 两者 paraphrase 波动区间都很宽，stable win 只有 14.08%/17.68%。
- large 的优势主要来自 Substitute；Add 仍明显弱于 Qwen、S2、LinCIR。

#### LinCIR / Pic2Word

- 两者原始 absolute retrieval 较弱（R@1 约 52%–53%），但 Delta 胜率仍达到
  80.53%/78.42%，说明它们更擅长相对组合区分，而不是在该候选池中取得最高绝对 rank。
- LinCIR 的 paraphrase 波动区间最窄，stable win 26.52%，优于 Qwen、Pic2Word 和
  BGE base/large。
- LinCIR 在 Add 上胜率最高（85.79%），尤其适合新增对象/属性类组合。
- Pic2Word 与 LinCIR 的胜负一致率 83.39%、Delta Spearman 0.765，属于高度相近但
  LinCIR 整体更强的一组。

### 3. Recall、NDCG 与正负检索集合变化

#### Positive query 的绝对检索

以下结果使用全部 500 条 positive query。每条 query 只有一个 binary relevant target。

| 模型 | R@1 | R@5 | R@10 | NDCG@10 | MRR |
|---|---:|---:|---:|---:|---:|
| BGE-VL-MLLM-S2 | **80.60%** | **97.00%** | **99.00%** | **0.9053** | **0.8778** |
| BGE-VL-MLLM-S1 | 74.20% | 95.80% | 97.80% | 0.8674 | 0.8323 |
| Qwen3-VL-Embedding-2B | 72.40% | 93.80% | 98.20% | 0.8569 | 0.8175 |
| BGE-VL-large | 71.20% | 93.20% | 97.20% | 0.8494 | 0.8109 |
| BGE-VL-base | 66.40% | 93.20% | 96.40% | 0.8271 | 0.7836 |
| Pic2Word | 53.00% | 82.40% | 91.80% | 0.7210 | 0.6627 |
| LinCIR | 52.20% | 85.00% | 93.40% | 0.7329 | 0.6715 |

NDCG 只有一个 binary relevant target，因此本质是带 rank discount 的 target-rank
指标，不是多级相关性 NDCG。

#### Strict negative 的 target hit 与 Top-10 Jaccard

为保证正负对照完全同口径，下表只使用有 Add/Substitute strict negative 的 493 个
pair。`q+ R@K` 是 positive query 的 target Recall；`q− Hit@K` 是故意改错后的 query
仍把原 target 放入 Top-K 的比例。Negative 行先在每个 pair 内平均，再跨 pair 平均，
避免 negative 数量较多的 pair 获得更大权重。在相同 positive Recall 下，`q− Hit@K`
越低，表示模型越能排斥错误修改。

| 模型 | q+ R@1 | q− Hit@1 | q+ R@5 | q− Hit@5 | q+ R@10 | q− Hit@10 | Top-10 Jaccard |
|---|---:|---:|---:|---:|---:|---:|---:|
| BGE-VL-MLLM-S2 | 80.53% | 70.07% | 96.96% | 93.62% | 98.99% | 97.35% | 0.6655 |
| BGE-VL-MLLM-S1 | 74.24% | 60.91% | 95.74% | 90.97% | 97.77% | 95.81% | 0.6490 |
| Qwen3-VL-Embedding-2B | 72.41% | 64.60% | 93.71% | 91.31% | 98.17% | 96.27% | **0.7189** |
| BGE-VL-large | 71.20% | 56.29% | 93.10% | 88.06% | 97.16% | 92.90% | 0.6416 |
| BGE-VL-base | 66.33% | 52.63% | 93.10% | 87.21% | 96.35% | 92.59% | **0.6331** |
| Pic2Word | 53.14% | 43.83% | 82.35% | 77.36% | 91.89% | 86.93% | 0.6379 |
| LinCIR | 52.33% | 44.62% | 84.99% | 79.79% | 93.31% | 89.25% | 0.6710 |

Jaccard 在每条 strict negative 上计算
`|Top10(q+) ∩ Top10(q−)| / |Top10(q+) ∪ Top10(q−)|`，再按 pair 平衡平均。它只比较
图片 ID 集合，不比较文本 token，也不考虑 Top-10 内部顺序。高 Jaccard 表示错误编辑
只局部改变候选集合，不表示 negative 检索正确。

| 模型 | q+ target rank 更好 | 正负 rank 相同 | 平均 rank 退化 | q+/q− cosine |
|---|---:|---:|---:|---:|
| BGE-VL-base | 30.98% | 61.17% | 1.373 | 0.8653 |
| BGE-VL-large | 30.31% | 63.37% | 1.422 | 0.8454 |
| BGE-VL-MLLM-S1 | 26.09% | 67.86% | 0.941 | 0.8795 |
| BGE-VL-MLLM-S2 | 19.65% | 75.99% | 0.639 | 0.8857 |
| Qwen3-VL-Embedding-2B | 19.82% | 74.44% | 0.668 | 0.9084 |
| LinCIR | **33.15%** | 56.89% | 1.941 | 0.9138 |
| Pic2Word | 32.45% | 56.21% | **1.989** | 0.9131 |

这里 `平均 rank 退化 = rank(q−) - rank(q+)`，正数表示错误修改让 target 排名变差。
S2/Qwen 的相同 rank 比例很高，是因为 493-target pool 较小，正负 query 经常都把
target 排在第 1；所以离散 rank 出现 ceiling effect，而连续 Delta 仍能发现 score
margin 的变化。

#### Delete 的描述性检索结果

Delete 只是移除信息，不保证和 target 冲突，因此不能当作 strict negative。下表在
457 个有 Delete 的 pair 上独立作 pair-balanced 描述。

| 模型 | q+ R@1 | Delete Hit@1 | q+ R@10 | Delete Hit@10 | Top-10 Jaccard |
|---|---:|---:|---:|---:|---:|
| BGE-VL-MLLM-S2 | 81.18% | 72.03% | 99.34% | 97.85% | 0.7076 |
| BGE-VL-MLLM-S1 | 75.49% | 64.33% | 97.81% | 96.32% | 0.6953 |
| Qwen3-VL-Embedding-2B | 72.87% | 61.27% | 98.03% | 96.54% | 0.7628 |
| BGE-VL-large | 71.12% | 61.85% | 97.37% | 95.73% | 0.7047 |
| BGE-VL-base | 67.40% | 60.69% | 96.50% | 94.71% | 0.6973 |
| Pic2Word | 53.83% | 47.85% | 92.12% | 88.44% | 0.6716 |
| LinCIR | 53.61% | 49.45% | 93.65% | 90.59% | 0.7274 |

### 4. CPU 5-fold grouped probe 全量结果

`python run_diagnostics.py --models all --probe-only` 已对 7 个模型全部完成：每个模型
使用 2,446 条 negative、500 个 `pair_id` group、5 个 outer fold；同一 pair 不会跨
train/test。分类器的 C 在每个 outer fold 内部调参，置信区间使用 1,000 次 bootstrap。
以下均为 out-of-fold accuracy。

#### 8 类线性 probe

| 模型 | Raw 9-cell | Direction 9-cell | Semantic | Operation | Norm-only 9-cell | q+ 9-cell | q− 9-cell | Text TF-IDF 9-cell |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BGE-VL-base | 56.54% | 62.14% | 58.30% | 72.89% | 15.82% | 12.14% | 32.38% | 27.15% |
| BGE-VL-large | 56.42% | 61.00% | 58.34% | 73.71% | 15.58% | 12.14% | 32.05% | 27.15% |
| BGE-VL-MLLM-S1 | **57.65%** | **64.47%** | 58.54% | 71.79% | 17.33% | **12.47%** | **33.40%** | 27.15% |
| BGE-VL-MLLM-S2 | 57.15% | 63.16% | 58.01% | 72.81% | **22.12%** | 12.43% | 33.32% | 27.15% |
| Qwen3-VL-Embedding-2B | 49.96% | 54.70% | **58.71%** | 65.66% | 21.26% | 11.24% | 29.11% | 27.15% |
| LinCIR | 54.54% | 60.92% | 58.59% | **73.75%** | 20.03% | 12.06% | 30.62% | 27.15% |
| Pic2Word | 53.56% | 59.53% | 57.77% | **73.75%** | 19.99% | 11.73% | 29.35% | 27.15% |
| 均匀机会 | 11.11% | 11.11% | 33.33% | 33.33% | 11.11% | 11.11% | 11.11% | 11.11% |

- Direction 普遍优于 raw vector，说明 3×3 cell 信息主要编码在编辑方向，而不是向量
  长度；Norm-only 明显较弱也支持这一点。
- Operation 达 65.66%–73.75%，明显高于 Semantic 的 57.77%–58.71%，说明操作类型
  比 object/attribute/predicate 更容易线性分离。
- q+ 接近机会水平，q− 与 TF-IDF 高于机会水平；标签信号主要来自负文本及正负差向量，
  不是原始 positive query 的 pair 身份。
- TF-IDF 对所有模型完全相同是正常的，因为 7 个模型使用同一批文本。它反映数据集的
  词面泄漏基线，不是模型指标。
- Probe 衡量表征几何，不是检索质量。Qwen 的 Delta/检索很强，但 direction probe
  最低，不能据此判定 Qwen 性能差。

#### 每模型 3×3 Delta 胜率

该表来自 `probe-only` 刷新的 `delta_by_cell.csv`；其中 Delete 仍只作描述。列名使用
`O/A/P = object/attribute/predicate`，`Add/Del/Sub = add/delete/substitute`。

| 模型 | O-Add | O-Del | O-Sub | A-Add | A-Del | A-Sub | P-Add | P-Del | P-Sub |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BGE-VL-base | 56.67% | 76.60% | 88.21% | 73.23% | 62.13% | 79.07% | 65.84% | 65.07% | 65.61% |
| BGE-VL-large | 68.06% | 77.30% | **92.05%** | 78.79% | 57.10% | 78.74% | 68.32% | 61.03% | 69.12% |
| BGE-VL-MLLM-S1 | 80.56% | 75.89% | 88.97% | 77.27% | 73.96% | 80.73% | 50.93% | **77.57%** | 73.68% |
| BGE-VL-MLLM-S2 | 90.56% | **85.82%** | 88.46% | 76.26% | **75.74%** | **84.39%** | 59.63% | 76.84% | 72.28% |
| Qwen3-VL-Embedding-2B | **91.67%** | 76.60% | 90.51% | 82.83% | 65.09% | 80.07% | 72.05% | 66.18% | **74.04%** |
| LinCIR | **91.67%** | 56.03% | 88.97% | **83.84%** | 55.03% | 74.09% | **72.67%** | 50.00% | 63.16% |
| Pic2Word | 86.67% | 63.83% | 90.77% | 76.77% | 59.47% | 73.75% | 68.94% | 51.10% | 59.65% |

#### Factorial structure

Factor model 检查 edit vector 是否可以只由 semantic 主效应和 operation 主效应解释。
Saturated 允许每个 3×3 cell 有独立中心；Interaction gain 是 saturated R² 减 additive
R²。全部都是 grouped held-out 结果。

| 模型 | Additive R² | Saturated 9-cell R² | Interaction gain | Additive prediction cosine | Saturated prediction cosine |
|---|---:|---:|---:|---:|---:|
| BGE-VL-base | 0.0371 | **0.0518** | **0.0147** | **0.1729** | **0.2060** |
| BGE-VL-large | 0.0275 | 0.0402 | 0.0127 | 0.1529 | 0.1863 |
| BGE-VL-MLLM-S1 | 0.0267 | 0.0389 | 0.0122 | 0.1376 | 0.1705 |
| BGE-VL-MLLM-S2 | 0.0358 | 0.0496 | 0.0138 | 0.1494 | 0.1791 |
| Qwen3-VL-Embedding-2B | 0.0129 | 0.0203 | 0.0074 | 0.1015 | 0.1275 |
| LinCIR | 0.0289 | 0.0386 | 0.0097 | 0.1491 | 0.1734 |
| Pic2Word | 0.0259 | 0.0348 | 0.0089 | 0.1412 | 0.1630 |

R² 只有 1.29%–5.18%，表示 3×3 标签只能解释 edit vector 的很小一部分；样本没有
坍缩成 9 个固定模板。Interaction gain 也仅 0.74%–1.47%，没有强烈的 cell-centroid
交互结构。

#### Reference ablation（strict Add + Substitute）

`q_reference = f_query(reference, empty text)`。Local alignment 衡量从 reference-only
query 移向组合 query 时，局部移动是否朝向 target；下表按 negative 数量加权。所有模型
的 positive alignment 都高于 negative，说明正确修改平均更朝向 target。

| 模型 | Positive local alignment | Negative local alignment | 差值 | 正负 shift direction cosine |
|---|---:|---:|---:|---:|
| BGE-VL-base | 0.1785 | 0.1501 | 0.0284 | 0.9057 |
| BGE-VL-large | 0.1943 | 0.1604 | 0.0339 | 0.8931 |
| BGE-VL-MLLM-S1 | 0.3110 | 0.2812 | 0.0298 | 0.8612 |
| BGE-VL-MLLM-S2 | 0.2381 | 0.2089 | 0.0291 | 0.8525 |
| Qwen3-VL-Embedding-2B | **0.3540** | 0.3008 | **0.0532** | 0.8345 |
| LinCIR | 0.1183 | 0.1028 | 0.0155 | 0.8707 |
| Pic2Word | 0.1770 | 0.1617 | 0.0153 | **0.9244** |

正负 shift cosine 均较高（0.835–0.924），说明正确/错误文本相对 reference 产生的
总体移动方向仍高度相似；真正区分它们的是沿 target 方向的细小差异。不同模型空间的
cosine 标尺不同，因此 alignment 原始值不用于跨模型性能排名。

CPU 明细保存在每个模型的 `analysis/` 下，包括 `probe_metrics.csv`、
`delta_by_cell.csv`、`reference_ablation_by_cell.csv`、`factorial_report.json`、各 probe
的 confusion matrix 和逐条 out-of-fold prediction；跨模型 probe 汇总为
`comparison/probe_metrics_all_models.csv`。

### 5. 模型间一致性

- S1/S2：胜负一致率 85.63%，Delta Spearman 0.793。
- LinCIR/Pic2Word：83.39%，0.765。
- BGE base/large：81.04%，0.785。
- 跨系列通常只有约 70%–76% 一致率，说明困难样本具有模型族特异性，存在 ensemble
  或按错误类型路由的空间。

## 三、建议

1. 立即删除或重新生成 2 条 no-op Delete，并在 finalization 增加
   `negative_text.casefold() != positive_text.casefold()` 硬校验。
2. 主训练和主排名使用 Add + Substitute；Delete 单独采样、单独加权和单独报告。
3. 定向补充 predicate Add/Substitute 和 attribute Add，缓解 cell 数量与审核选择偏差。
4. 增加独立 validator/人工抽检，并在下一版加入 vision-grounded construction。
5. 当前所有数字来自 train-500，只能作为训练集诊断；正式模型选择应建立独立 held-out
   split。
6. 若需要标准 CIRR 6-image group Recall@1/2/3，仍需为 2,239 张尚未缓存的 group
   candidate 图像补算每个模型的 candidate embedding。现有 target-pool 结果不能冒充
   标准 CIRR Recall。
