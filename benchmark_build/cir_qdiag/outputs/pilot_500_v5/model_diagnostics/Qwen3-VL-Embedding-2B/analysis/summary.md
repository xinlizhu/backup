# Qwen3-VL-Embedding-2B 3×3 diagnostic

- Samples: 459
- Negatives: 3401
- Embedding dimension: 2048
- All probe splits are grouped by pair_id.

## Linear probes

| Probe | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| raw_v__cell | 0.4587 | 0.4413 | 0.4621 |
| raw_v__semantic_target | 0.5792 | 0.5880 | 0.5756 |
| raw_v__edit_operation | 0.6536 | 0.6489 | 0.6764 |
| direction_v__cell | 0.5413 | 0.5200 | 0.5353 |
| norm_v__cell | 0.1870 | 0.1103 | 0.1529 |
| q_positive__cell | 0.1256 | 0.1231 | 0.1393 |
| q_negative__cell | 0.2952 | 0.2830 | 0.2972 |
| raw_v__construction_mode | 0.7016 | 0.6706 | 0.7152 |
| raw_v__cell__text_only | 0.4780 | 0.4633 | 0.4842 |
| raw_v__cell__vision_grounded | 0.3787 | 0.2731 | 0.3380 |
| text_tfidf__cell | 0.3073 | 0.2844 | 0.2947 |
| raw_v__semantic_target__text_only_editable_to_vision_grounded | 0.4906 | 0.3851 | 0.3971 |
| raw_v__semantic_target__vision_grounded_to_text_only_editable | 0.3307 | 0.2888 | 0.3230 |
| raw_v__edit_operation__text_only_editable_to_vision_grounded | 0.4718 | 0.4516 | 0.5242 |
| raw_v__edit_operation__vision_grounded_to_text_only_editable | 0.5432 | 0.5296 | 0.5615 |

## Factorial structure

- Additive held-out R²: 0.0148
- Saturated 9-cell held-out R²: 0.0214
- Interaction gain: 0.0066

Delete negatives are weaker queries and are not required to contradict the target; 
interpret their delta descriptively rather than as strict retrieval errors.
