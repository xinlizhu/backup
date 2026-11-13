# BGE-VL-large 3×3 diagnostic

- Samples: 500
- Negatives: 2446
- Embedding dimension: 768
- All probe splits are grouped by pair_id.
- Reference ablation f_query(R, empty) is available in reference_ablation_by_cell.csv.

## Linear probes

| Probe | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| raw_v__cell | 0.5642 | 0.5522 | 0.5725 |
| raw_v__semantic_target | 0.5834 | 0.5834 | 0.5827 |
| raw_v__edit_operation | 0.7371 | 0.7423 | 0.7473 |
| direction_v__cell | 0.6100 | 0.6027 | 0.6184 |
| norm_v__cell | 0.1558 | 0.1096 | 0.1506 |
| q_positive__cell | 0.1214 | 0.1190 | 0.1299 |
| q_negative__cell | 0.3205 | 0.3003 | 0.3248 |
| text_tfidf__cell | 0.2715 | 0.2569 | 0.2733 |

## Factorial structure

- Additive held-out R²: 0.0275
- Saturated 9-cell held-out R²: 0.0402
- Interaction gain: 0.0127

Delete negatives are weaker queries and are not required to contradict the target; 
interpret their delta descriptively rather than as strict retrieval errors.
