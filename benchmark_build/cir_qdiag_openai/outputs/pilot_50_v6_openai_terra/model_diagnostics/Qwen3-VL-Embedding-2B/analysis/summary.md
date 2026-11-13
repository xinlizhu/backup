# Qwen3-VL-Embedding-2B 3×3 diagnostic

- Samples: 48
- Negatives: 331
- Embedding dimension: 2048
- All probe splits are grouped by pair_id.
- Reference ablation f_query(R, empty) is available in reference_ablation_by_cell.csv.

## Linear probes

| Probe | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| raw_v__cell | 0.3958 | 0.3558 | 0.3602 |
| raw_v__semantic_target | 0.5680 | 0.5645 | 0.5641 |
| raw_v__edit_operation | 0.5740 | 0.5624 | 0.5981 |
| direction_v__cell | 0.4350 | 0.4055 | 0.3937 |
| norm_v__cell | 0.2115 | 0.1450 | 0.1860 |
| q_positive__cell | 0.0967 | 0.0915 | 0.1026 |
| q_negative__cell | 0.2477 | 0.2088 | 0.2244 |
| raw_v__construction_mode | 0.5952 | 0.5932 | 0.5975 |
| raw_v__cell__text_only | 0.4235 | 0.4094 | 0.4109 |
| raw_v__cell__vision_grounded | 0.4472 | 0.3333 | 0.3335 |
| text_tfidf__cell | 0.2326 | 0.1805 | 0.1977 |
| raw_v__semantic_target__text_only_editable_to_vision_grounded | 0.4783 | 0.4578 | 0.4605 |
| raw_v__semantic_target__vision_grounded_to_text_only_editable | 0.5176 | 0.5273 | 0.5332 |
| raw_v__edit_operation__text_only_editable_to_vision_grounded | 0.3665 | 0.3446 | 0.4414 |
| raw_v__edit_operation__vision_grounded_to_text_only_editable | 0.4235 | 0.3882 | 0.4330 |

## Factorial structure

- Additive held-out R²: 0.0029
- Saturated 9-cell held-out R²: -0.0019
- Interaction gain: -0.0048

Delete negatives are weaker queries and are not required to contradict the target; 
interpret their delta descriptively rather than as strict retrieval errors.
