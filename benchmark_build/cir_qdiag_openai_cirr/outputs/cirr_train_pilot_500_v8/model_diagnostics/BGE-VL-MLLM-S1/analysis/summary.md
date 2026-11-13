# BGE-VL-MLLM-S1 3×3 diagnostic

- Samples: 500
- Negatives: 2446
- Embedding dimension: 4096
- All probe splits are grouped by pair_id.
- Reference ablation f_query(R, empty) is available in reference_ablation_by_cell.csv.

## Linear probes

| Probe | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| raw_v__cell | 0.5765 | 0.5606 | 0.5825 |
| raw_v__semantic_target | 0.5854 | 0.5875 | 0.5856 |
| raw_v__edit_operation | 0.7179 | 0.7245 | 0.7288 |
| direction_v__cell | 0.6447 | 0.6367 | 0.6512 |
| norm_v__cell | 0.1733 | 0.1241 | 0.1585 |
| q_positive__cell | 0.1247 | 0.1223 | 0.1339 |
| q_negative__cell | 0.3340 | 0.3064 | 0.3328 |
| text_tfidf__cell | 0.2715 | 0.2569 | 0.2733 |

## Factorial structure

- Additive held-out R²: 0.0267
- Saturated 9-cell held-out R²: 0.0389
- Interaction gain: 0.0122

Delete negatives are weaker queries and are not required to contradict the target; 
interpret their delta descriptively rather than as strict retrieval errors.
