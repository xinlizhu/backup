# BGE-VL-MLLM-S2 3×3 diagnostic

- Samples: 500
- Negatives: 2446
- Embedding dimension: 4096
- All probe splits are grouped by pair_id.
- Reference ablation f_query(R, empty) is available in reference_ablation_by_cell.csv.

## Linear probes

| Probe | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| raw_v__cell | 0.5715 | 0.5562 | 0.5788 |
| raw_v__semantic_target | 0.5801 | 0.5832 | 0.5801 |
| raw_v__edit_operation | 0.7281 | 0.7346 | 0.7365 |
| direction_v__cell | 0.6316 | 0.6234 | 0.6370 |
| norm_v__cell | 0.2212 | 0.1409 | 0.1845 |
| q_positive__cell | 0.1243 | 0.1223 | 0.1372 |
| q_negative__cell | 0.3332 | 0.3081 | 0.3355 |
| text_tfidf__cell | 0.2715 | 0.2569 | 0.2733 |

## Factorial structure

- Additive held-out R²: 0.0358
- Saturated 9-cell held-out R²: 0.0496
- Interaction gain: 0.0138

Delete negatives are weaker queries and are not required to contradict the target; 
interpret their delta descriptively rather than as strict retrieval errors.
