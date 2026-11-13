# Pic2Word 3×3 diagnostic

- Samples: 500
- Negatives: 2446
- Embedding dimension: 768
- All probe splits are grouped by pair_id.
- Reference ablation f_query(R, empty) is available in reference_ablation_by_cell.csv.

## Linear probes

| Probe | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| raw_v__cell | 0.5356 | 0.5145 | 0.5443 |
| raw_v__semantic_target | 0.5777 | 0.5787 | 0.5782 |
| raw_v__edit_operation | 0.7375 | 0.7415 | 0.7509 |
| direction_v__cell | 0.5953 | 0.5814 | 0.6008 |
| norm_v__cell | 0.1999 | 0.1358 | 0.1733 |
| q_positive__cell | 0.1173 | 0.1158 | 0.1327 |
| q_negative__cell | 0.2935 | 0.2752 | 0.3002 |
| text_tfidf__cell | 0.2715 | 0.2569 | 0.2733 |

## Factorial structure

- Additive held-out R²: 0.0259
- Saturated 9-cell held-out R²: 0.0348
- Interaction gain: 0.0089

Delete negatives are weaker queries and are not required to contradict the target; 
interpret their delta descriptively rather than as strict retrieval errors.
