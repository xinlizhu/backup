# Full baseline vs Full Hard-0.5：纯统计分析

本报告只使用原始检索、直接 response 统计、跨模型 q+ rerank 与 Top-K 集合比较。
没有训练或汇总 OOF LogisticRegression/learned reranker。

## 1. 官方全量 PinPoint

| 模型 | mAP@10 | NegRecall@10 |
| --- | ---: | ---: |
| Full baseline | 0.067656 | 0.171946 |
| Full Hard-0.5 | 0.067326 | 0.163631 |
| Hard − baseline | -0.000330 | -0.008315 |

Hard-0.5 的官方 mAP@10 略降 0.000330，但 NegRecall@10 降低 0.008315（约
4.84% 相对下降）。因此 hard training 的主要收益是减少 harmful/official-negative
候选进入前十，而不是提高整体正例排序质量。

## 2. Negative-text sensitivity

| 模型 | mean target Delta | raw win rate |
| --- | ---: | ---: |
| Full baseline | 0.030563 | 71.98% |
| Full Hard-0.5 | 0.071304 | 80.46% |

Hard-0.5 明显增强了 positive text 相对生成 q- 的方向分离；但这个提升没有等比例转化为
官方 mAP。这说明 synthetic text discrimination 与真实图库中的多正例检索不是同一个问题。

## 3. 原始 Top-K Recall、nDCG 与候选 Jaccard（官方 7,635 queries）

| K | 候选交集/K | Jaccard | baseline PosRecall | hard PosRecall | Δ PosRecall | baseline NegRecall | hard NegRecall | Δ NegRecall | baseline nDCG | hard nDCG | Δ nDCG |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.8584 | 0.8584 | 0.03838 | 0.04833 | +0.00995 | 0.46680 | 0.44820 | -0.01860 | 0.03838 | 0.04833 | +0.00995 |
| 5 | 0.7215 | 0.5982 | 0.12708 | 0.12740 | +0.00033 | 0.23020 | 0.21836 | -0.01185 | 0.10922 | 0.11065 | +0.00144 |
| 10 | 0.7261 | 0.5927 | 0.14584 | 0.14309 | -0.00275 | 0.17195 | 0.16363 | -0.00832 | 0.12142 | 0.12052 | -0.00091 |
| 20 | 0.7323 | 0.5948 | 0.18756 | 0.18293 | -0.00464 | 0.13180 | 0.12518 | -0.00662 | 0.14065 | 0.13864 | -0.00201 |
| 50 | 0.7443 | 0.6060 | 0.26998 | 0.26196 | -0.00802 | 0.13023 | 0.12312 | -0.00711 | 0.17374 | 0.17052 | -0.00322 |

结论很清楚：Hard-0.5 改善了 Top-1，并在所有 K 上减少 official negatives；但从
Top-10 开始，positive coverage 逐渐低于 baseline，Top-20/50 的 nDCG 也略低。它更像是在
“收紧头部、增强拒绝”，而不是提高整个候选列表的正例覆盖。

两模型并非检索出完全不同的图库：Top-10 平均共享 7.26/10 个候选，Top-50 共享
37.22/50 个候选。约四分之一候选发生变化，足以改变 recall，但不是表示空间完全重构。

## 4. Response profile 的关键现象

在各自 native Top-50 中，只看已经被检索到的 positives：

| 模型 | retrieved positive − pool nonpositive 的 D_available |
| --- | ---: |
| Full baseline | 0.010577 |
| Full Hard-0.5 | 0.012439 |

在 shared-union 固定候选上，Hard 相对 baseline 的该 separation 增量为
`+0.002868`，95% CI `[0.001472, 0.004264]`。因此 Hard 确实加强了“已经能检索到的
positive”与当前竞争者之间的 response separation。

但把 Top-50 外追加的所有官方 positives 也算进来时，结论反转：Hard 相对 baseline 的
`all positive − pool nonpositive` separation 变化为 `-0.003477`，95% CI
`[-0.004627, -0.002303]`。此外：

| 模型 | retrieved positive − appended positive 的 D_available |
| --- | ---: |
| Full baseline | 0.015333 |
| Full Hard-0.5 | 0.023964 |

Hard 将这个“容易检索的 positive 与漏检 positive”差距进一步扩大了约 `+0.008019`。
这解释了为什么 Delta/胜率和 retrieved-positive separation 都变强，完整 Positive Recall
却没有同步提高：新增能力主要集中在模型本来就容易处理的 positive 上。

validated positive 与同 query 的 unseen official positive 之间没有明显 D_available 差异：
baseline 为 `0.000858`（CI `[-0.001163, 0.002975]`），Hard 为 `0.000720`
（CI `[-0.001847, 0.003601]`）。这项审计没有发现明显的 target-specific 差异，但不能消除
q- 生成阶段看过 target 的总体限制。

## 5. Baseline / Hard 互相 rerank（只使用正常 q+ score）

实验严格分成两个方向：retriever 从完整图库召回自己的 native Top-50；reranker 用自己的
positive query embedding 和自己的 candidate image embedding 计算 cosine score，在这 50 个
候选中重排。完全不使用生成 negative text。

### 官方全量 7,635 queries

| Retriever → Reranker | native mAP | rerank mAP | ΔmAP | native NegRecall | rerank NegRecall | ΔNegRecall | Top-10 Jaccard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline → Hard-0.5 | 0.067656 | 0.067512 | -0.000144 | 0.171946 | 0.164194 | -0.007752 | 0.6004 |
| Hard-0.5 → baseline | 0.067326 | 0.067623 | +0.000296 | 0.163631 | 0.169920 | +0.006289 | 0.6048 |

方向性非常清楚：

- Hard 作为 reranker 会继承其 negative 抑制能力，显著降低 baseline pool 的 NegRecall，但
  mAP/Positive Recall 略降。
- baseline 作为 reranker 会恢复一些正例排序和 mAP，但会抵消 Hard retriever 的 negative
  抑制优势。

因此两个模型的 score 确实互补，但互补表现为“baseline 更偏正例覆盖，Hard 更偏 negative
抑制”，目前没有一个方向同时改善 mAP 和 NegRecall。

在 strict 1,580 queries 上，baseline → Hard 的 group-macro NegRecall 变化为 `-0.006773`
（CI `[-0.011396, -0.001977]`），是稳定的；mAP 变化 CI 跨 0。Hard → baseline 的 mAP、
Positive Recall、nDCG 和 NegRecall 的 group-macro CI 都跨 0。

Top-50 Jaccard 等于 1，因为候选池不变；Top-10 Jaccard 约 0.60，说明换一个模型的 q+ score
会明显改变哪些候选进入前十。

## 总结

1. Hard-0.5 成功学会更强的 negative-text 抑制，并改善 Top-1。
2. 它稳定减少 official negatives，但 Top-10/20/50 的 positive coverage 没有改善，深层
   Recall 与 nDCG 反而略降。
3. response profile 对“已检索到的 easy positives”有效，对漏检 positives 的迁移不足。
4. baseline → Hard cross-rerank 能稳定降低 NegRecall，但没有稳定 mAP 增益；反方向会牺牲
   Hard 的 negative 抑制。
5. 下一步应继续做无机器学习实验：按 available-category 数量与 edit 类别分层，以及分析
   baseline-only/hard-only Top-K 候选的 positive/official-negative/unlabeled 构成。
