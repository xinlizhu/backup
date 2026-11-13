# Counterfactual response profile experiment

## 口径

- 官方 PinPoint 全量 **7,635 queries** 是主检索基准，报告原版 evaluator 的 mAP@10 与 NegRecall@10。
- V7 的 **500 groups / 1933 queries** 是官方 query 的子集，仅它们有生成并审核过的反事实 negative text，因此用于 response profile、probe 和诊断性 rerank。
- 两者不是二选一：全量回答“模型最终检索怎样”，500 组子集回答“模型为何这样，以及对 negative edit 如何响应”。

## 决策总表

| model | analysis_mode | pair_balanced_mean_target_delta | pair_balanced_target_raw_win_rate | official_full_native_mAP@10 | official_full_native_NegRecall@10 | pilot_strict_mAP@10::native | pilot_strict_NegRecall@10::native |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CLIP-L-full-baseline | simple_no_ml | 0.030563 | 0.719765 | 0.067656 | 0.171946 | 0.065705 | 0.190513 |
| CLIP-L-full-hard-w0.5 | simple_no_ml | 0.071304 | 0.804601 | 0.067326 | 0.163631 | 0.066725 | 0.183228 |

本次按 simple_no_ml 口径运行：没有训练 LogisticRegression，也没有使用 learned reranker；只报告原始检索、直接统计和跨模型 q+ similarity rerank。



## 稀疏 negative 规则

- strict 主分析只使用 ADD / SUBSTITUTE；DELETE 不进入主 response。
- 每条 query 有多少 accepted negative 就用多少：先对同一类别内文本求均值，再对该 query 实际存在的类别等权平均。
- 缺失类别保持 NaN，并带显式 mask；不填 0，不要求六类齐全。
- 有至少一类 strict negative 的 query：**1580**；完整 6D 仅 **5**，只报告覆盖率，不据此训练 probe。
- 固定 object_add + object_substitute 二维 cohort 是预先指定的补充验证。

## 防止误读的控制

- 所有官方 positives 都作为 multi-positive 标签；Top-K 外 positive 只追加到诊断训练集，不加入 rerank 池。
- 官方 negatives 与未标注 Top-K competitors 分开；未标注候选不被称作语义错误。
- 本次关闭全部 OOF probe/learned reranker；旧运行遗留文件不进入本报告汇总。
- positive paraphrase ensemble、mask-only、candidate-shuffle、category-permutation 均作为 placebo/control。
- category-permutation 固定原有 missingness pattern，只在该 query 已出现的 edit 类别之间重分配 response；其 value 与显式命名的 permuted mask 列始终配套，mask 数值本身不随机置换。
- shared-union pool 用于 baseline/hard 模型的固定候选公平比较；native pool 用于真实运行指标。
- negative-response separation 仍是 target-informed 诊断；跨模型 rerank 完全不读取 V7 negative text，只使用两个现有模型的正常 q+ embedding。
- cross-model rerank 固定使用 retriever 的 native Top-50；reranker 只提供自己的正常 q+ cosine score，不读取 negative text。

## 候选构建

| model | pool_name | query_count | candidate_row_count | retrieved_pool_row_count | appended_positive_row_count |
| --- | --- | --- | --- | --- | --- |
| CLIP-L-full-baseline | native_top50 | 1933 | 108495 | 96650 | 11845 |
| CLIP-L-full-hard-w0.5 | native_top50 | 1933 | 108527 | 96650 | 11877 |
| CLIP-L-full-baseline | shared_union::CLIP-L-full-baseline::CLIP-L-full-hard-w0.5 | 1933 | 132239 | 120813 | 11426 |
| CLIP-L-full-hard-w0.5 | shared_union::CLIP-L-full-baseline::CLIP-L-full-hard-w0.5 | 1933 | 132239 | 120813 | 11426 |

## 原有 positive-vs-negative-text Delta / 胜率（同一批 q- 现场复算）

| model | subset | group_count | instruction_count | negative_count | pair_balanced_mean_target_delta | mean_target_delta_ci_low | mean_target_delta_ci_high | pair_balanced_target_raw_win_rate | target_raw_win_rate_ci_low | target_raw_win_rate_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CLIP-L-full-baseline | strict | 471 | 1580 | 3242 | 0.030563 | 0.027747 | 0.033592 | 0.719765 | 0.702187 | 0.737095 |
| CLIP-L-full-hard-w0.5 | strict | 471 | 1580 | 3242 | 0.071304 | 0.066334 | 0.076574 | 0.804601 | 0.787362 | 0.820674 |

## Target exposure 泄漏审计

V7 q- 的生成/审核看过 validated positives。下表在同一 query 内比较这些 positives 与生成阶段
未列入 validated set 的其它官方 positives；它能暴露明显的 target-specific response，但不能替代
真正 target-blind 的独立测试集。

| model | dimension | first_mean | second_mean | estimate | ci_low | ci_high | query_count | group_count | holm_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CLIP-L-full-baseline | d_available | 0.033083 | 0.032225 | 0.000858 | -0.001163 | 0.002975 | 1066 | 322 | 1.000000 |
| CLIP-L-full-baseline | d_object_add | 0.024497 | 0.022622 | 0.001875 | -0.000296 | 0.004032 | 631 | 264 | 1.000000 |
| CLIP-L-full-baseline | d_object_substitute | 0.039634 | 0.039206 | 0.000428 | -0.002490 | 0.003323 | 503 | 226 | 1.000000 |
| CLIP-L-full-baseline | d_attribute_add | 0.022345 | 0.022605 | -0.000260 | -0.003387 | 0.002824 | 232 | 152 | 1.000000 |
| CLIP-L-full-baseline | d_attribute_substitute | 0.039477 | 0.038931 | 0.000547 | -0.004058 | 0.005339 | 380 | 158 | 1.000000 |
| CLIP-L-full-baseline | d_predicate_add | 0.013092 | 0.012740 | 0.000353 | -0.002542 | 0.003811 | 160 | 106 | 1.000000 |
| CLIP-L-full-baseline | d_predicate_substitute | 0.007718 | 0.009554 | -0.001835 | -0.004454 | 0.000398 | 195 | 119 | 1.000000 |
| CLIP-L-full-hard-w0.5 | d_available | 0.073305 | 0.072585 | 0.000720 | -0.001847 | 0.003601 | 1066 | 322 | 1.000000 |
| CLIP-L-full-hard-w0.5 | d_object_add | 0.092512 | 0.091744 | 0.000767 | -0.002129 | 0.003553 | 631 | 264 | 1.000000 |
| CLIP-L-full-hard-w0.5 | d_object_substitute | 0.059109 | 0.059178 | -0.000069 | -0.003082 | 0.002942 | 503 | 226 | 1.000000 |
| CLIP-L-full-hard-w0.5 | d_attribute_add | 0.083545 | 0.084399 | -0.000853 | -0.005809 | 0.004006 | 232 | 152 | 1.000000 |
| CLIP-L-full-hard-w0.5 | d_attribute_substitute | 0.053697 | 0.054159 | -0.000463 | -0.007121 | 0.006454 | 380 | 158 | 1.000000 |
| CLIP-L-full-hard-w0.5 | d_predicate_add | 0.068925 | 0.068780 | 0.000145 | -0.004769 | 0.005496 | 160 | 106 | 1.000000 |
| CLIP-L-full-hard-w0.5 | d_predicate_substitute | 0.016650 | 0.018619 | -0.001969 | -0.004926 | 0.000626 | 195 | 119 | 1.000000 |

## 机器学习诊断

本次默认关闭 OOF LogisticRegression、AUROC/AUPRC probe 和 learned reranker。


## Native 检索核心结果

| model | variant | cohort | query_count | mAP@10 | delta_mAP@10 | NegRecall@10 | delta_NegRecall@10 | Top1JaccardVsNative | Top5JaccardVsNative | Top10JaccardVsNative | Top20JaccardVsNative | Top50JaccardVsNative | rerank_eligible_query_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CLIP-L-full-baseline | native | official_full_native_7635 | 7635 | 0.067656 | 0.000000 | 0.171946 | 0.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 |
| CLIP-L-full-baseline | native | pilot_strict_negative_eligible | 1580 | 0.065705 | 0.000000 | 0.190513 | 0.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 |
| CLIP-L-full-hard-w0.5 | native | official_full_native_7635 | 7635 | 0.067326 | 0.000000 | 0.163631 | 0.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 |
| CLIP-L-full-hard-w0.5 | native | pilot_strict_negative_eligible | 1580 | 0.066725 | 0.000000 | 0.183228 | 0.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 |

## 跨模型 Top-50 rerank（不使用 negative text）

`retriever_model` 先从完整图库召回自己的 native Top-50；`reranker_model` 使用自己的
正常 positive query embedding 与自己的 candidate image embedding 计算 cosine score，只在
这 50 个 candidate 内重新排序。

| retriever_model | reranker_model | cohort | query_count | native_mAP@10 | mAP@10 | delta_mAP@10 | native_PositiveRecall@10 | PositiveRecall@10 | delta_PositiveRecall@10 | native_nDCG@10 | nDCG@10 | delta_nDCG@10 | native_NegRecall@10 | NegRecall@10 | delta_NegRecall@10 | Top1JaccardVsNative | Top5JaccardVsNative | Top10JaccardVsNative | Top20JaccardVsNative | Top50JaccardVsNative | uses_negative_text |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CLIP-L-full-baseline | CLIP-L-full-hard-w0.5 | official_full_7635 | 7635 | 0.067656 | 0.067512 | -0.000144 | 0.145839 | 0.143881 | -0.001958 | 0.121422 | 0.120970 | -0.000452 | 0.171946 | 0.164194 | -0.007752 | 0.859201 | 0.600718 | 0.600363 | 0.622457 | 1.000000 | False |
| CLIP-L-full-baseline | CLIP-L-full-hard-w0.5 | pilot_500_all_queries | 1933 | 0.063184 | 0.064958 | 0.001774 | 0.148815 | 0.145253 | -0.003562 | 0.118418 | 0.118608 | 0.000190 | 0.189288 | 0.180540 | -0.008748 | 0.870150 | 0.610126 | 0.608279 | 0.626466 | 1.000000 | False |
| CLIP-L-full-baseline | CLIP-L-full-hard-w0.5 | pilot_strict_negative_eligible | 1580 | 0.065705 | 0.066838 | 0.001134 | 0.152369 | 0.148591 | -0.003778 | 0.122628 | 0.122128 | -0.000500 | 0.190513 | 0.183102 | -0.007411 | 0.869620 | 0.616428 | 0.611351 | 0.629702 | 1.000000 | False |
| CLIP-L-full-baseline | CLIP-L-full-hard-w0.5 | pilot_object_add_substitute_pair | 442 | 0.051021 | 0.052765 | 0.001743 | 0.125253 | 0.124053 | -0.001200 | 0.093837 | 0.094824 | 0.000987 | 0.182191 | 0.175856 | -0.006335 | 0.898190 | 0.606667 | 0.610853 | 0.630089 | 1.000000 | False |
| CLIP-L-full-hard-w0.5 | CLIP-L-full-baseline | official_full_7635 | 7635 | 0.067326 | 0.067623 | 0.000296 | 0.143090 | 0.144881 | 0.001791 | 0.120515 | 0.121083 | 0.000568 | 0.163631 | 0.169920 | 0.006289 | 0.860511 | 0.603250 | 0.604754 | 0.630118 | 1.000000 | False |
| CLIP-L-full-hard-w0.5 | CLIP-L-full-baseline | pilot_500_all_queries | 1933 | 0.064896 | 0.063954 | -0.000941 | 0.145228 | 0.147940 | 0.002712 | 0.118615 | 0.118768 | 0.000152 | 0.180644 | 0.186563 | 0.005920 | 0.871702 | 0.614962 | 0.613656 | 0.635004 | 1.000000 | False |
| CLIP-L-full-hard-w0.5 | CLIP-L-full-baseline | pilot_strict_negative_eligible | 1580 | 0.066725 | 0.066558 | -0.000166 | 0.148434 | 0.151442 | 0.003008 | 0.122054 | 0.123036 | 0.000983 | 0.183228 | 0.187749 | 0.004521 | 0.870886 | 0.620632 | 0.616318 | 0.637158 | 1.000000 | False |
| CLIP-L-full-hard-w0.5 | CLIP-L-full-baseline | pilot_object_add_substitute_pair | 442 | 0.052678 | 0.052463 | -0.000216 | 0.123471 | 0.124612 | 0.001141 | 0.094695 | 0.094871 | 0.000176 | 0.176988 | 0.178571 | 0.001584 | 0.900452 | 0.608903 | 0.616260 | 0.638714 | 1.000000 | False |

## Baseline / Hard 原始检索集合 Jaccard

Jaccard 只比较 Top-K 候选集合，不考虑集合内部顺序；同时报告更直观的
`mean_overlap_fraction = |A ∩ B| / K`。

| baseline_model | hard_model | cohort | k | query_count | mean_jaccard | mean_overlap_fraction | baseline_positive_recall | hard_positive_recall | delta_positive_recall | baseline_positive_hit | hard_positive_hit | delta_positive_hit | baseline_negative_recall | hard_negative_recall | delta_negative_recall | baseline_ndcg | hard_ndcg | delta_ndcg | bootstrap_unit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CLIP-L-full-baseline | CLIP-L-full-hard-w0.5 | official_full_7635 | 1 | 7635 | 0.858415 | 0.858415 | 0.038376 | 0.048330 | 0.009954 | 0.038376 | 0.048330 | 0.009954 | 0.466798 | 0.448199 | -0.018599 | 0.038376 | 0.048330 | 0.009954 | query |
| CLIP-L-full-baseline | CLIP-L-full-hard-w0.5 | official_full_7635 | 5 | 7635 | 0.598202 | 0.721467 | 0.127077 | 0.127404 | 0.000327 | 0.314604 | 0.310936 | -0.003667 | 0.230203 | 0.218356 | -0.011847 | 0.109216 | 0.110653 | 0.001438 | query |
| CLIP-L-full-baseline | CLIP-L-full-hard-w0.5 | official_full_7635 | 10 | 7635 | 0.592695 | 0.726051 | 0.145839 | 0.143090 | -0.002749 | 0.406942 | 0.399607 | -0.007335 | 0.171946 | 0.163631 | -0.008315 | 0.121422 | 0.120515 | -0.000907 | query |
| CLIP-L-full-baseline | CLIP-L-full-hard-w0.5 | official_full_7635 | 20 | 7635 | 0.594774 | 0.732331 | 0.187563 | 0.182928 | -0.004635 | 0.485920 | 0.473870 | -0.012050 | 0.131801 | 0.125182 | -0.006619 | 0.140645 | 0.138635 | -0.002010 | query |
| CLIP-L-full-baseline | CLIP-L-full-hard-w0.5 | official_full_7635 | 50 | 7635 | 0.606046 | 0.744309 | 0.269977 | 0.261955 | -0.008022 | 0.585986 | 0.573674 | -0.012312 | 0.130230 | 0.123117 | -0.007113 | 0.173739 | 0.170516 | -0.003223 | query |
| CLIP-L-full-baseline | CLIP-L-full-hard-w0.5 | pilot_500_all_queries | 1 | 1933 | 0.858267 | 0.870150 | 0.029488 | 0.043456 | 0.011867 | 0.029488 | 0.043456 | 0.011867 | 0.501811 | 0.477496 | -0.018100 | 0.029488 | 0.043456 | 0.011867 | v7_group |
| CLIP-L-full-baseline | CLIP-L-full-hard-w0.5 | pilot_500_all_queries | 5 | 1933 | 0.600777 | 0.729850 | 0.122125 | 0.124840 | 0.000677 | 0.309364 | 0.299017 | -0.011600 | 0.253440 | 0.243094 | -0.012553 | 0.102029 | 0.106099 | 0.001540 | v7_group |
| CLIP-L-full-baseline | CLIP-L-full-hard-w0.5 | pilot_500_all_queries | 10 | 1933 | 0.594837 | 0.733833 | 0.148815 | 0.145228 | -0.004286 | 0.410760 | 0.398862 | -0.010633 | 0.189288 | 0.180644 | -0.008199 | 0.118418 | 0.118615 | -0.001429 | v7_group |
| CLIP-L-full-baseline | CLIP-L-full-hard-w0.5 | pilot_500_all_queries | 20 | 1933 | 0.591210 | 0.737791 | 0.200748 | 0.194200 | -0.005649 | 0.502845 | 0.486808 | -0.013133 | 0.146274 | 0.137623 | -0.008849 | 0.142122 | 0.140780 | -0.002495 | v7_group |
| CLIP-L-full-baseline | CLIP-L-full-hard-w0.5 | pilot_500_all_queries | 50 | 1933 | 0.602822 | 0.749995 | 0.280791 | 0.276735 | -0.004678 | 0.600103 | 0.591826 | -0.011167 | 0.142316 | 0.135252 | -0.007221 | 0.174233 | 0.173438 | -0.002057 | v7_group |
| CLIP-L-full-baseline | CLIP-L-full-hard-w0.5 | pilot_strict_negative_eligible | 1 | 1580 | 0.853857 | 0.869620 | 0.032278 | 0.044937 | 0.009837 | 0.032278 | 0.044937 | 0.009837 | 0.506962 | 0.483544 | -0.019285 | 0.032278 | 0.044937 | 0.009837 | v7_group |
| CLIP-L-full-baseline | CLIP-L-full-hard-w0.5 | pilot_strict_negative_eligible | 5 | 1580 | 0.608569 | 0.735063 | 0.126814 | 0.126255 | -0.002824 | 0.318987 | 0.305063 | -0.014968 | 0.256361 | 0.247120 | -0.010035 | 0.106372 | 0.108478 | -0.000995 | v7_group |
| CLIP-L-full-baseline | CLIP-L-full-hard-w0.5 | pilot_strict_negative_eligible | 10 | 1580 | 0.598134 | 0.736899 | 0.152369 | 0.148434 | -0.004405 | 0.417089 | 0.409494 | -0.005874 | 0.190513 | 0.183228 | -0.006784 | 0.122628 | 0.122054 | -0.002267 | v7_group |
| CLIP-L-full-baseline | CLIP-L-full-hard-w0.5 | pilot_strict_negative_eligible | 20 | 1580 | 0.594902 | 0.740127 | 0.206094 | 0.198658 | -0.003062 | 0.509494 | 0.489873 | -0.007714 | 0.148473 | 0.139858 | -0.009424 | 0.146771 | 0.144426 | -0.002138 | v7_group |
| CLIP-L-full-baseline | CLIP-L-full-hard-w0.5 | pilot_strict_negative_eligible | 50 | 1580 | 0.605061 | 0.751633 | 0.289180 | 0.283569 | -0.003519 | 0.601899 | 0.590506 | -0.010191 | 0.144713 | 0.137605 | -0.007505 | 0.179772 | 0.177740 | -0.002266 | v7_group |
| CLIP-L-full-baseline | CLIP-L-full-hard-w0.5 | pilot_object_add_substitute_pair | 1 | 442 | 0.869048 | 0.898190 | 0.015837 | 0.031674 | 0.016369 | 0.015837 | 0.031674 | 0.016369 | 0.470588 | 0.470588 | 0.018601 | 0.015837 | 0.031674 | 0.016369 | v7_group |
| CLIP-L-full-baseline | CLIP-L-full-hard-w0.5 | pilot_object_add_substitute_pair | 5 | 442 | 0.599037 | 0.729412 | 0.094796 | 0.098567 | 0.000114 | 0.233032 | 0.226244 | -0.006696 | 0.242081 | 0.233937 | -0.011443 | 0.077128 | 0.080511 | 0.000954 | v7_group |
| CLIP-L-full-baseline | CLIP-L-full-hard-w0.5 | pilot_object_add_substitute_pair | 10 | 442 | 0.608270 | 0.736652 | 0.125253 | 0.123471 | -0.001353 | 0.319005 | 0.319005 | -0.000372 | 0.182191 | 0.176988 | -0.007597 | 0.093837 | 0.094695 | 0.000329 | v7_group |
| CLIP-L-full-baseline | CLIP-L-full-hard-w0.5 | pilot_object_add_substitute_pair | 20 | 442 | 0.601257 | 0.741516 | 0.163681 | 0.158057 | -0.007111 | 0.384615 | 0.361991 | -0.035565 | 0.145397 | 0.139539 | -0.006607 | 0.110901 | 0.110496 | -0.002315 | v7_group |
| CLIP-L-full-baseline | CLIP-L-full-hard-w0.5 | pilot_object_add_substitute_pair | 50 | 442 | 0.612814 | 0.754208 | 0.231115 | 0.226629 | -0.008315 | 0.477376 | 0.461538 | -0.029985 | 0.137209 | 0.134322 | -0.004575 | 0.135560 | 0.135343 | -0.002646 | v7_group |

## 固定 shared-pool 模型差

| baseline_model | hard_model | contrast | dimension | estimate | ci_low | ci_high | query_count | group_count | holm_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CLIP-L-full-baseline | CLIP-L-full-hard-w0.5 | all_positive_minus_rank_violating_official_negative | d_add | -0.008019 | -0.010881 | -0.005484 | 893 | 335 | 0.011994 |
| CLIP-L-full-baseline | CLIP-L-full-hard-w0.5 | all_positive_minus_rank_violating_official_negative | d_attribute | -0.007146 | -0.010912 | -0.003906 | 639 | 276 | 0.011994 |
| CLIP-L-full-baseline | CLIP-L-full-hard-w0.5 | all_positive_minus_rank_violating_official_negative | d_attribute_add | -0.009050 | -0.014055 | -0.004952 | 338 | 193 | 0.011994 |
| CLIP-L-full-baseline | CLIP-L-full-hard-w0.5 | all_positive_minus_rank_violating_official_negative | d_attribute_substitute | -0.005491 | -0.010377 | -0.001407 | 418 | 197 | 0.047976 |
| CLIP-L-full-baseline | CLIP-L-full-hard-w0.5 | all_positive_minus_rank_violating_official_negative | d_available | -0.005897 | -0.008610 | -0.003555 | 1196 | 381 | 0.011994 |
| CLIP-L-full-baseline | CLIP-L-full-hard-w0.5 | all_positive_minus_rank_violating_official_negative | d_available_minus_abs_paraphrase | -0.000434 | -0.004475 | 0.003426 | 1196 | 381 | 1.000000 |
| CLIP-L-full-baseline | CLIP-L-full-hard-w0.5 | all_positive_minus_rank_violating_official_negative | d_available_shuffled | -0.003875 | -0.005332 | -0.002470 | 1196 | 381 | 0.011994 |
| CLIP-L-full-baseline | CLIP-L-full-hard-w0.5 | all_positive_minus_rank_violating_official_negative | d_category_macro_weighting_permuted | -0.005897 | -0.008674 | -0.003675 | 1196 | 381 | 0.011994 |
| CLIP-L-full-baseline | CLIP-L-full-hard-w0.5 | all_positive_minus_rank_violating_official_negative | d_edit_max | -0.012510 | -0.016027 | -0.009323 | 1196 | 381 | 0.011994 |
| CLIP-L-full-baseline | CLIP-L-full-hard-w0.5 | all_positive_minus_rank_violating_official_negative | d_edit_micro | -0.005897 | -0.008570 | -0.003641 | 1196 | 381 | 0.011994 |
| CLIP-L-full-baseline | CLIP-L-full-hard-w0.5 | all_positive_minus_rank_violating_official_negative | d_edit_min | -0.000075 | -0.003109 | 0.002553 | 1196 | 381 | 1.000000 |
| CLIP-L-full-baseline | CLIP-L-full-hard-w0.5 | all_positive_minus_rank_violating_official_negative | d_edit_std | -0.005570 | -0.007169 | -0.003886 | 1196 | 381 | 0.011994 |
| CLIP-L-full-baseline | CLIP-L-full-hard-w0.5 | all_positive_minus_rank_violating_official_negative | d_group_available | -0.005206 | -0.008116 | -0.002811 | 1196 | 381 | 0.011994 |
| CLIP-L-full-baseline | CLIP-L-full-hard-w0.5 | all_positive_minus_rank_violating_official_negative | d_max | -0.012510 | -0.015941 | -0.009252 | 1196 | 381 | 0.011994 |
| CLIP-L-full-baseline | CLIP-L-full-hard-w0.5 | all_positive_minus_rank_violating_official_negative | d_min | -0.000075 | -0.003095 | 0.002550 | 1196 | 381 | 1.000000 |
| CLIP-L-full-baseline | CLIP-L-full-hard-w0.5 | all_positive_minus_rank_violating_official_negative | d_object | -0.005001 | -0.007089 | -0.002919 | 961 | 355 | 0.011994 |
| CLIP-L-full-baseline | CLIP-L-full-hard-w0.5 | all_positive_minus_rank_violating_official_negative | d_object_add | -0.006815 | -0.009589 | -0.004142 | 722 | 302 | 0.011994 |
| CLIP-L-full-baseline | CLIP-L-full-hard-w0.5 | all_positive_minus_rank_violating_official_negative | d_object_substitute | -0.002660 | -0.004910 | -0.000446 | 586 | 262 | 0.119940 |
| CLIP-L-full-baseline | CLIP-L-full-hard-w0.5 | all_positive_minus_rank_violating_official_negative | d_predicate | -0.004694 | -0.007811 | -0.001724 | 406 | 216 | 0.011994 |
| CLIP-L-full-baseline | CLIP-L-full-hard-w0.5 | all_positive_minus_rank_violating_official_negative | d_predicate_add | -0.011251 | -0.017260 | -0.005467 | 199 | 132 | 0.011994 |

跨模型模型差仅报告直接 response separation 与原始 Top-K Jaccard；不计算 OOF/DID。

## 输出说明

- `negative_text_coverage_*.csv/json`：每条 query 的稀疏维度覆盖率。
- `target_exposure_audit_all_models.csv`：validated 与 unseen official positives 的同-query response 敏感性。
- `native_ranking_jaccard_summary.csv`：baseline/hard 原始 Top-K 候选集合重合度。
- `native_topk_retrieval_summary.csv`：同表的 Recall/HitRate/NegRecall/nDCG 导向别名。
- `cross_model_rerank_results.csv`：baseline/hard 互相作为 retriever/reranker 的正式结果。
- `cross_model_rerank_by_query.parquet`：跨模型 rerank 的逐 query 指标与 Jaccard。
- `native_ranking_jaccard_by_query.csv`：逐 query、逐 K 的 overlap 与 Jaccard。
- `PAIR_DECISION_SUMMARY.csv`：shared-union 固定候选上的预注册 profile 增量 DID 主表。
- `<model>/candidate_profiles_native.parquet`：candidate-level response，不把缺失值补 0。
- `<model>/negative_text_sensitivity_summary.csv`：原有 target Delta / win rate 基线。
- `<model>/profile_dimension_stats.csv`：multi-positive 候选分离与 group bootstrap CI。
- `<model>/probe_results.csv`：按 group 的 OOF probe 和 placebo。
- `<model>/paired_feature_probe_differences.csv`：严格相同 cohort 上 sim→sim+D 的配对增益。
- `<model>/rerank/rerank_metrics_all_cohorts.csv`：全量、Pilot、eligible、固定二维 cohort。
- `shared_pool_comparisons/`：固定候选并集的模型配对比较。
- `shared_pool_incremental_feature_did.csv`：profile 相对 sim 的增量在 Hard 与 baseline 间的配对 DID。
