# PinPoint V7 Delta mode analysis

PinPoint V7 does not contain the old benchmark's paired `text_only_editable/vision_grounded` construction modes. Every V7 candidate is text-only generated and then vision-grounded selected. This directory keeps the historical name `delta_mode_analysis`, but analyzes model, edit operation, 3×3 cell, and Positive-review scope.

The histogram variable is each accepted negative's mean target Delta:

```text
mean_t[sim(f(R,m+),t) - sim(f(R,m-),t)]
```

Empirical zoom used by plots: `[-0.15, 0.35]`.

## Strict overview

| Model | tau | q+ score | q- score | mean Delta | raw target win | robust target win | Delta NDCG@10 | Jaccard@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BGE-VL-base | 0.1173 | 0.2101 | 0.1952 | 0.0149 | 60.77% | 3.37% | 0.0739 | 0.2764 |
| BGE-VL-MLLM-S1 | 0.0918 | 0.2470 | 0.2237 | 0.0233 | 69.97% | 8.26% | 0.0924 | 0.2944 |

## Win-rate heatmaps

- [BGE-base raw/robust 3×3 heatmaps](delta_win_rate_heatmap_BGE-VL-base.png)
- [MLLM-S1 raw/robust 3×3 heatmaps](delta_win_rate_heatmap_BGE-VL-MLLM-S1.png)

Notes:

- Full-range histograms use the theoretical cosine-difference interval `[-2,2]`; zoom plots show the actual V7 concentration.
- Histogram counts are accepted-negative rows. Summary means and confidence intervals are hierarchical group-balanced statistics.
- The red threshold line is model-specific positive-paraphrase Q95.
- DELETE is a weaker-query diagnostic and should not be interpreted as a strict hard-negative success rate.
- Review-scope plots are descriptive because fully and partially reviewed groups are not content matched.
