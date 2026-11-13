我想新增一个诊断实验，不训练任何新模型，只分析现有 CIR checkpoint 的 negative-text sensitivity 是否真的能区分真实 positive image 和 retrieval hard-negative images。

> **最终实现说明（2026-08-07）**
>
> 本文下面保留的是最初实验构想；可运行协议已经实现于
> `../diag_model/counterfactual_profile_analysis.py`。最终实现对几个会导致偏差的初始假设做了修正：
>
> **2026-08-10 更新：默认协议已改为 `simple_no_ml`。下面原构想中的 OOF
> LogisticRegression/probe 仅作为历史设计保留，默认运行不会训练或汇总它们。当前只看直接
> separation、Delta/胜率、官方 mAP@10/NegRecall@10、跨模型 q+ rerank 和 Top-K Jaccard。
> rerank 不使用生成 negative text。**
>
> 1. 官方 PinPoint 全量 7,635 queries 用于 headline mAP@10 / NegRecall@10；V7 500 groups
>    （1,933 个官方 query 的子集）只提供反事实 negative text，用于机制诊断。二者联合使用，
>    不是二选一。
> 2. 不要求每条 query 都有六类 negative。主特征先在类别内平均，再对实际存在的类别等权
>    平均；缺失单元保持 NaN 并带 mask。完整 6D 仅 5 条 query，不能作为主 probe cohort。
> 3. CV group 使用真实 V7 `group_id`，不是 `query_id`；同一 reference/positive set 的多条
>    paraphrase 不会跨 train/test。
> 4. 标签分成 official positive、official negative、unlabeled Top-50 competitor。代码同时报告
>    clean official-negative probe 与 benchmark-nonpositive probe，但不会把 unlabeled candidate
>    称为人工语义 negative。
> 5. baseline/hard 除各自 native Top-50 外，还在两者 Top-50 union 的固定候选集上重新跑相同
>    OOF probe，并对模型差做 paired group bootstrap。
> 6. 主 learned rerank 使用候选池内标签的 OOF prediction；Top-K 外追加 positive 只用于
>    augmented 机制诊断，不进入 rerank 池。全量 rerank 行是 pilot-only hybrid，非 Pilot query
>    保持原排序。
> 7. 跨模型主证据不是直接横比各自 native Top-50 的 probe，而是在相同 shared-union 候选、
>    相同 query/group 上计算
>    `[(sim + profile) - sim]_Hard - [(sim + profile) - sim]_baseline` 的 paired bootstrap DID。
>    rank-violating cohort 由模型自身排序决定，只用于模型内部诊断，不做伪配对比较。
>
> 推荐运行：
>
> ```bash
> cd /mnt/data/datasets/search_dataset/zxl/benchmark_build/diag_model
> CUDA_VISIBLE_DEVICES=0 bash run_counterfactual_profile_analysis.sh
> ```
>
> 详细口径和输出见 `../diag_model/README.md` 顶部章节。

## 目标

对于每一个 CIR query：

```text
(reference_image, positive_modify_text, positive_target)
```

我们已经有 6 类 harmful negative modification text：

```text
object_add
object_substitute
attribute_add
attribute_substitute
predicate_add
predicate_substitute
```

不使用 DELETE。

我希望分析：

> 对真实 retrieval Top-K candidates，每张 candidate image 对这 6 种 counterfactual modification 的响应是否不同；这种 6 维 response profile 能否区分 positive target 和真实 hard-negative candidates。

---

## 1. 使用哪些模型

先对以下已有模型做诊断：

```text
Full baseline
Full Hard-0.5
```

如果代码支持方便，再扩展：

```text
Full Hard-0.1
Full Hard-0.25
```

不要重新训练模型。

优先复用 PinPoint 官方全量评估已经缓存的 candidate image embeddings 和 positive query embeddings。

每个模型必须使用自己对应 checkpoint 编码得到的 image/query embedding，不允许跨模型共享 embedding。

---

## 2. 每个 query 获取真实 Top-K candidates

对于每个官方 PinPoint query：

```text
q+ = F(reference_image, positive_modify_text)
```

使用当前模型在完整 candidate gallery 中正常检索。

保存 Top-K candidates，建议：

```text
K = 50
```

同时至少分析：

```text
Top-10
Top-20
Top-50
```

对于每个 candidate，需要保存：

```text
query_id
candidate_id
original_rank
is_positive
is_annotated_negative（如果 PinPoint 有这个标注）
sim_positive_query
```

其中：

```text
sim_positive_query = sim(q+, z_j)
```

---

## 3. 编码 6 个 negative queries

对同一个 reference image，分别编码：

```text
q_neg[0] = object_add
q_neg[1] = object_substitute
q_neg[2] = attribute_add
q_neg[3] = attribute_substitute
q_neg[4] = predicate_add
q_neg[5] = predicate_substitute
```

如果一个 query 某一类有多条 negative text，不要直接把它们混成独立 query。

先对每条 negative 分别计算结果，然后在同一个 negative category 内求平均。

例如 object_add 有 3 条 negative：

```text
D_object_add(candidate)
=
mean_n [
    sim(q+, candidate)
    -
    sim(q_object_add_n, candidate)
]
```

最终每个 candidate 得到一个固定 6 维向量。

---

## 4. 定义 Counterfactual Response Profile

对于 candidate image (z_j)，对每一类 negative (c) 定义：

```text
D_j[c] =
sim(q+, z_j)
-
mean_neg_in_class(sim(q_neg[c], z_j))
```

因此每个 candidate 得到：

```text
D_j = [
    D_object_add,
    D_object_substitute,
    D_attribute_add,
    D_attribute_substitute,
    D_predicate_add,
    D_predicate_substitute
]
```

称为：

```text
Counterfactual Response Profile
```

同时保存：

```text
D_mean = mean(D_j)
D_max
D_min
D_std
```

注意不要只统计 target image。

我要的是每个 query 的真实 Top-K candidates 全部计算这个 profile。

---

## 5. 最核心的比较

对于每个 query，把 candidates 分成：

### A. Positive target

```text
positive target image
```

### B. Hard negatives

重点使用：

```text
由 q+ 实际检索出来的 Top-K 非 positive candidates
```

如果 PinPoint 有官方 negative annotation，再额外区分：

```text
official_negative
unlabeled_topK_negative
```

不要把整个图库随机 negative 当作主要比较对象。

我们最关心真正会影响 ranking 的 hard negatives。

---

## 6. 第一组统计：Positive vs Hard Negative 的 D 是否不同

分别统计：

```text
mean(D_positive[c])
mean(D_hard_negative[c])
difference
bootstrap 95% CI
```

对六类分别报告。

输出类似：

| response dimension | positive mean | hard-neg mean | difference | 95% CI |
| ------------------ | ------------: | ------------: | ---------: | -----: |
| object-add         |               |               |            |        |
| object-sub         |               |               |            |        |
| attribute-add      |               |               |            |        |
| attribute-sub      |               |               |            |        |
| predicate-add      |               |               |            |        |
| predicate-sub      |               |               |            |        |

注意 bootstrap 最好以 query/group 为单位，而不是把所有 candidate 当独立样本。

---

## 7. 第二组统计：6D profile 能不能区分 positive / hard negative

这是最关键的实验。

构造样本：

```text
X = 6D Counterfactual Response Profile
y = positive / hard-negative
```

训练一个非常简单的 probe：

```text
Logistic Regression
```

不要训练 MLP。

必须 query-disjoint split：

```text
同一个 query 的 candidates 不能同时出现在 train 和 test。
```

建议：

```text
5-fold group cross validation
group = query_id
```

报告：

```text
AUROC
AUPRC
accuracy
balanced accuracy
```

然后做以下 feature ablation：

```text
1. 仅 sim(q+, candidate)
2. 仅 D_mean
3. 6D D profile
4. sim(q+, candidate) + 6D D profile
```

这里最重要的是比较：

```text
sim(q+, candidate)
vs.
sim(q+, candidate) + response profile
```

如果加入 6D profile 后 AUROC/AUPRC 明显提升，说明 negative-text sensitivity 包含原始 retrieval score 没有利用到的 candidate discrimination 信息。

---

## 8. 第三组：直接测试 reranking potential，但先不训练复杂模型

使用 Logistic Regression 得到：

```text
P(candidate_is_positive | sim_positive, D_1...D_6)
```

把这个 probability 作为 candidate score，仅在原始 Top-K 中 rerank。

分别测试：

```text
Top-10
Top-20
Top-50
```

计算 PinPoint 官方：

```text
mAP@10
NegRecall@10
```

这里不要重新训练 CIR encoder，只是诊断 response profile 是否包含可利用的 ranking information。

注意 group split / out-of-fold prediction，不能在同一批 query 上训练 probe 后再直接评估，否则会数据泄漏。

最好使用 5-fold query-group CV：

* 4 folds train logistic regression
* 1 fold 生成 reranking score
* 合并全部 out-of-fold predictions
* 最后统一跑官方 evaluator

---

## 9. 第四组：Baseline vs Hard 模型比较

对于 Full baseline 和 Full Hard-0.5，分别计算：

```text
Positive/negative profile separation
6D probe AUROC
sim + 6D probe AUROC
out-of-fold reranking mAP@10
out-of-fold NegRecall@10
```

我要回答的问题是：

```text
Hard-0.5 虽然 negative-text Delta / win rate 更高，
它的 Counterfactual Response Profile 是否也更能区分
real positive target 和 real retrieval hard negatives？
```

理想情况下输出一个总表：

| Model         | Text Delta | Text WinRate | 6D AUROC | sim-only AUROC | sim+6D AUROC | rerank mAP@10 | rerank NegRecall@10 |
| ------------- | ---------: | -----------: | -------: | -------------: | -----------: | ------------: | ------------------: |
| Full baseline |            |              |          |                |              |               |                     |
| Full Hard-0.5 |            |              |          |                |              |               |                     |

已有 Text Delta / WinRate 可以直接读取现有结果，不必重新算也可以。

---

## 10. 再做一个非常重要的 ranking-level 统计

对于每个 query 的 positive target 和当前排名高于它的 candidates：

假设 positive target rank = r。

把：

```text
rank < r
```

的错误 candidate 称为：

```text
rank-violating negatives
```

比较：

```text
D_positive
vs
D_rank_violating_negative
```

这比普通 Top-K negative 更重要，因为这些 candidate 才是真正导致 Recall@1 / Recall@K 失败的图片。

单独报告：

```text
positive vs rank-violating-negative probe AUROC
```

以及六个维度的平均差异。

---

## 11. 输出文件

希望至少生成：

```text
counterfactual_candidate_profiles.csv
```

每行一个 query-candidate pair：

```text
model
query_id
candidate_id
original_rank
is_positive
is_official_negative
sim_positive
D_object_add
D_object_substitute
D_attribute_add
D_attribute_substitute
D_predicate_add
D_predicate_substitute
D_mean
D_std
```

以及：

```text
profile_summary.json
profile_dimension_stats.csv
profile_probe_results.csv
profile_rerank_results.csv
```

再输出一份：

```text
COUNTERFACTUAL_PROFILE_ANALYSIS.md
```

把主要结果、表格和结论整理进去。

---

## 12. 我最终希望通过这个实验判断三种情况

### 情况 A

```text
6D profile 能明显区分 positive / hard negative，
并且 sim + 6D 明显优于 sim-only。
```

说明 negative text 学到的能力对真实 CIR candidate ranking 有信息，只是现有 cosine score 没有把它利用起来。

### 情况 B

```text
Full Hard 的 6D separation 比 baseline 强，
但加入 profile 后仍不能改善 real candidate discrimination。
```

说明 hard training 强化的主要是 synthetic negative-text discrimination，没有很好迁移到真实 image ranking。

### 情况 C

```text
positive 和 hard-negative 的 6D profile 基本不可分。
```

说明当前构造的 negative text 更适合作为 diagnostic benchmark，而不是直接作为提升 CIR ranking 的训练信号。

优先把这个实验实现成独立诊断脚本，不要修改当前训练 pipeline。
