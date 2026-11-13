# BGE-VL-base 3×3 diagnostic

- Samples: 500
- Negatives: 2446
- Embedding dimension: 512
- All probe splits are grouped by pair_id.
- Reference ablation f_query(R, empty) is available in reference_ablation_by_cell.csv.

## Linear probes

| Probe | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| raw_v__cell | 0.5654 | 0.5518 | 0.5776 |
| raw_v__semantic_target | 0.5830 | 0.5834 | 0.5828 |
| raw_v__edit_operation | 0.7289 | 0.7347 | 0.7398 |
| direction_v__cell | 0.6214 | 0.6146 | 0.6328 |
| norm_v__cell | 0.1582 | 0.1152 | 0.1524 |
| q_positive__cell | 0.1214 | 0.1188 | 0.1327 |
| q_negative__cell | 0.3238 | 0.3031 | 0.3289 |
| text_tfidf__cell | 0.2715 | 0.2569 | 0.2733 |

## Factorial structure

- Additive held-out R²: 0.0371
- Saturated 9-cell held-out R²: 0.0518
- Interaction gain: 0.0147

Delete negatives are weaker queries and are not required to contradict the target; 
interpret their delta descriptively rather than as strict retrieval errors.
