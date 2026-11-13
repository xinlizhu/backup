# BGE-VL-MLLM-S1 3×3 diagnostic

- Samples: 48
- Negatives: 331
- Embedding dimension: 4096
- All probe splits are grouped by pair_id.
- Reference ablation f_query(R, empty) is available in reference_ablation_by_cell.csv.

## Linear probes

| Probe | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| raw_v__cell | 0.5770 | 0.5239 | 0.5461 |
| raw_v__semantic_target | 0.5831 | 0.5856 | 0.5797 |
| raw_v__edit_operation | 0.6888 | 0.6738 | 0.7039 |
| direction_v__cell | 0.5770 | 0.5251 | 0.5351 |
| norm_v__cell | 0.1541 | 0.1271 | 0.1518 |
| q_positive__cell | 0.1178 | 0.1089 | 0.1164 |
| q_negative__cell | 0.3112 | 0.2657 | 0.2946 |
| raw_v__construction_mode | 0.5952 | 0.5950 | 0.5950 |
| raw_v__cell__text_only | 0.5529 | 0.5376 | 0.5669 |
| raw_v__cell__vision_grounded | 0.5093 | 0.3793 | 0.3992 |
| text_tfidf__cell | 0.2326 | 0.1805 | 0.1977 |
| raw_v__semantic_target__text_only_editable_to_vision_grounded | 0.4472 | 0.4280 | 0.4307 |
| raw_v__semantic_target__vision_grounded_to_text_only_editable | 0.4882 | 0.4901 | 0.5132 |
| raw_v__edit_operation__text_only_editable_to_vision_grounded | 0.5155 | 0.4751 | 0.6039 |
| raw_v__edit_operation__vision_grounded_to_text_only_editable | 0.5176 | 0.5050 | 0.5396 |

## Factorial structure

- Additive held-out R²: 0.0130
- Saturated 9-cell held-out R²: 0.0159
- Interaction gain: 0.0029

Delete negatives are weaker queries and are not required to contradict the target; 
interpret their delta descriptively rather than as strict retrieval errors.
