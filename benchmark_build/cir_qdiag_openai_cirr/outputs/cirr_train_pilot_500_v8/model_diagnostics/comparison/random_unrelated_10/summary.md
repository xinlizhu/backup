# Random-unrelated target margin

Each positive query is compared with 10 unique target images drawn without replacement from other pairs. Its own target and reference image are excluded. The shared draw seed is 42, and identical draws are used for every model.

`Margin = cosine(query, positive target) - mean cosine(query, 10 unrelated targets)`.

| Model | Positive score | Unrelated score | Mean margin | Median margin | P05 | Beats each draw | Beats all 10 | Stable-win threshold | Margin − threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BGE-VL-base | 0.3450 | 0.0298 | 0.3152 | 0.3216 | 0.1717 | 99.62% | 97.40% | 0.0760 | 0.2392 |
| BGE-VL-large | 0.3457 | 0.0274 | 0.3184 | 0.3247 | 0.1729 | 99.66% | 98.00% | 0.0746 | 0.2438 |
| BGE-VL-MLLM-S1 | 0.3726 | 0.0438 | 0.3288 | 0.3295 | 0.1991 | 99.84% | 98.40% | 0.0399 | 0.2889 |
| BGE-VL-MLLM-S2 | 0.3509 | 0.0391 | 0.3118 | 0.3147 | 0.2028 | 99.86% | 98.60% | 0.0371 | 0.2747 |
| Qwen3-VL-Embedding-2B | 0.6297 | 0.1525 | 0.4772 | 0.4828 | 0.2637 | 99.92% | 99.20% | 0.0724 | 0.4049 |
| LinCIR | 0.2675 | 0.0867 | 0.1808 | 0.1854 | 0.0945 | 99.56% | 96.60% | 0.0277 | 0.1532 |
| Pic2Word | 0.2666 | 0.1021 | 0.1644 | 0.1686 | 0.0782 | 99.48% | 95.40% | 0.0345 | 0.1300 |

The stable-win threshold is shown only as a scale reference. It was calibrated for positive-vs-generated-hard-negative Delta; the random unrelated margin is an easier background-separation diagnostic and is not itself a stable-win rate.
