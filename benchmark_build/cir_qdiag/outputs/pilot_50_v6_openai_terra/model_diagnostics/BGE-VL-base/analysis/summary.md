# BGE-VL-base 3×3 diagnostic

- Samples: 48
- Negatives: 331
- Embedding dimension: 512
- All probe splits are grouped by pair_id.
- Reference ablation f_query(R, empty) is available in reference_ablation_by_cell.csv.

## Linear probes

| Probe | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| raw_v__cell | 0.5166 | 0.4636 | 0.5010 |
| raw_v__semantic_target | 0.5982 | 0.5970 | 0.5954 |
| raw_v__edit_operation | 0.6465 | 0.6308 | 0.6486 |
| direction_v__cell | 0.5619 | 0.5282 | 0.5339 |
| norm_v__cell | 0.1903 | 0.1298 | 0.1811 |
| q_positive__cell | 0.1390 | 0.1294 | 0.1464 |
| q_negative__cell | 0.2779 | 0.2306 | 0.2572 |
| raw_v__construction_mode | 0.6042 | 0.6041 | 0.6052 |
| raw_v__cell__text_only | 0.5294 | 0.4856 | 0.5106 |
| raw_v__cell__vision_grounded | 0.4907 | 0.3933 | 0.4107 |
| text_tfidf__cell | 0.2326 | 0.1805 | 0.1977 |
| raw_v__semantic_target__text_only_editable_to_vision_grounded | 0.4596 | 0.4561 | 0.4543 |
| raw_v__semantic_target__vision_grounded_to_text_only_editable | 0.4882 | 0.4853 | 0.5164 |
| raw_v__edit_operation__text_only_editable_to_vision_grounded | 0.5466 | 0.4911 | 0.5634 |
| raw_v__edit_operation__vision_grounded_to_text_only_editable | 0.5412 | 0.5367 | 0.5544 |

## Factorial structure

- Additive held-out R²: 0.0213
- Saturated 9-cell held-out R²: 0.0265
- Interaction gain: 0.0051

Delete negatives are weaker queries and are not required to contradict the target; 
interpret their delta descriptively rather than as strict retrieval errors.
