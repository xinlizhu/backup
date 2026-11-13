# Train-500 target-pool retrieval diagnostic

This is not standard CIRR group/full-gallery Recall. The gallery contains 493 unique target images from the 500 diagnostic records.

Positive Recall/NDCG/MRR uses all 500 positive queries. Strict-negative metrics use Add + Substitute and are pair-balanced over the 493 covered pairs. A negative target hit means the intentionally incorrect query still retrieves the original target, so lower is preferable when positive Recall is held fixed.

| Model | Positive R@1 | Strict-negative hit@1 | Positive R@5 | Strict-negative hit@5 | Positive R@10 | Strict-negative hit@10 | NDCG@10 | MRR | Jaccard@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BGE-VL-MLLM-S2 | 0.8060 | 0.7007 | 0.9700 | 0.9362 | 0.9900 | 0.9735 | 0.9053 | 0.8778 | 0.6655 |
| Qwen3-VL-Embedding-2B | 0.7240 | 0.6460 | 0.9380 | 0.9131 | 0.9820 | 0.9627 | 0.8569 | 0.8175 | 0.7189 |
| BGE-VL-MLLM-S1 | 0.7420 | 0.6091 | 0.9580 | 0.9097 | 0.9780 | 0.9581 | 0.8674 | 0.8323 | 0.6490 |
| BGE-VL-large | 0.7120 | 0.5629 | 0.9320 | 0.8806 | 0.9720 | 0.9290 | 0.8494 | 0.8109 | 0.6416 |
| BGE-VL-base | 0.6640 | 0.5263 | 0.9320 | 0.8721 | 0.9640 | 0.9259 | 0.8271 | 0.7836 | 0.6331 |
| LinCIR | 0.5220 | 0.4462 | 0.8500 | 0.7979 | 0.9340 | 0.8925 | 0.7329 | 0.6715 | 0.6710 |
| Pic2Word | 0.5300 | 0.4383 | 0.8240 | 0.7736 | 0.9180 | 0.8693 | 0.7210 | 0.6627 | 0.6379 |

| Model | Positive better target rank | Equal target rank | Mean rank degradation | Query cosine |
|---|---:|---:|---:|---:|
| BGE-VL-MLLM-S2 | 0.1965 | 0.7599 | 0.6389 | 0.8857 |
| Qwen3-VL-Embedding-2B | 0.1982 | 0.7444 | 0.6676 | 0.9084 |
| BGE-VL-MLLM-S1 | 0.2609 | 0.6786 | 0.9412 | 0.8795 |
| BGE-VL-large | 0.3031 | 0.6337 | 1.4216 | 0.8454 |
| BGE-VL-base | 0.3098 | 0.6117 | 1.3726 | 0.8653 |
| LinCIR | 0.3315 | 0.5689 | 1.9411 | 0.9138 |
| Pic2Word | 0.3245 | 0.5621 | 1.9889 | 0.9131 |

NDCG uses one binary relevant target per query; it is therefore a discounted target-rank metric rather than graded relevance.
Jaccard@10 compares the positive and negative Top-10 image-ID sets; it does not use text tokens and does not measure correctness.
