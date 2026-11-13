# Training-set multi-model diagnostic

Completed models: 7

## Strict hard negatives (Add + Substitute)

| Rank | Model | Pair-balanced win rate | Mean delta | 95% CI (delta) |
|---:|---|---:|---:|---:|
| 1 | Qwen3-VL-Embedding-2B | 0.8373 | 0.0416 | [0.0390, 0.0440] |
| 2 | BGE-VL-MLLM-S2 | 0.8172 | 0.0292 | [0.0275, 0.0310] |
| 3 | LinCIR | 0.8053 | 0.0178 | [0.0167, 0.0190] |
| 4 | BGE-VL-MLLM-S1 | 0.7867 | 0.0271 | [0.0250, 0.0292] |
| 5 | Pic2Word | 0.7842 | 0.0170 | [0.0158, 0.0182] |
| 6 | BGE-VL-large | 0.7742 | 0.0329 | [0.0304, 0.0354] |
| 7 | BGE-VL-base | 0.7278 | 0.0276 | [0.0253, 0.0301] |

## Delete (reported separately)

Delete removes information from the positive instruction and is not required to contradict the target, so it is descriptive rather than part of the primary hard-negative ranking.

| Model | Pair-balanced win rate | Mean delta |
|---|---:|---:|
| BGE-VL-base | 0.6601 | 0.0192 |
| BGE-VL-large | 0.6185 | 0.0155 |
| BGE-VL-MLLM-S1 | 0.7586 | 0.0308 |
| BGE-VL-MLLM-S2 | 0.7702 | 0.0228 |
| Qwen3-VL-Embedding-2B | 0.6743 | 0.0194 |
| LinCIR | 0.5244 | 0.0024 |
| Pic2Word | 0.5635 | 0.0036 |

Raw cosine margins from different embedding spaces are not directly calibrated. Prefer win rate, sign agreement, and within-model geometry; treat cross-model delta differences as descriptive.
