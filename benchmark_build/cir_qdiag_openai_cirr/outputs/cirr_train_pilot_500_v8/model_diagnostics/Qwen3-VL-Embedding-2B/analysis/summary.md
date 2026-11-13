# Qwen3-VL-Embedding-2B 3×3 diagnostic

- Samples: 500
- Negatives: 2446
- Embedding dimension: 2048
- All probe splits are grouped by pair_id.
- Reference ablation f_query(R, empty) is available in reference_ablation_by_cell.csv.

## Linear probes

| Probe | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| raw_v__cell | 0.4996 | 0.4810 | 0.5042 |
| raw_v__semantic_target | 0.5871 | 0.5893 | 0.5875 |
| raw_v__edit_operation | 0.6566 | 0.6616 | 0.6728 |
| direction_v__cell | 0.5470 | 0.5301 | 0.5442 |
| norm_v__cell | 0.2126 | 0.1426 | 0.1809 |
| q_positive__cell | 0.1124 | 0.1119 | 0.1243 |
| q_negative__cell | 0.2911 | 0.2680 | 0.2871 |
| text_tfidf__cell | 0.2715 | 0.2569 | 0.2733 |

## Factorial structure

- Additive held-out R²: 0.0129
- Saturated 9-cell held-out R²: 0.0203
- Interaction gain: 0.0074

Delete negatives are weaker queries and are not required to contradict the target; 
interpret their delta descriptively rather than as strict retrieval errors.
