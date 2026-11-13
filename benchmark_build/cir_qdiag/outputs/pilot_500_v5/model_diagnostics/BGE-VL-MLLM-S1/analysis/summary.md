# BGE-VL-MLLM-S1 3×3 diagnostic

- Samples: 459
- Negatives: 3401
- Embedding dimension: 4096
- All probe splits are grouped by pair_id.

## Linear probes

| Probe | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| raw_v__cell | 0.5531 | 0.5348 | 0.5490 |
| raw_v__semantic_target | 0.5963 | 0.6009 | 0.5906 |
| raw_v__edit_operation | 0.7327 | 0.7306 | 0.7491 |
| direction_v__cell | 0.6272 | 0.6049 | 0.6160 |
| norm_v__cell | 0.1508 | 0.0987 | 0.1336 |
| q_positive__cell | 0.1320 | 0.1291 | 0.1482 |
| q_negative__cell | 0.3481 | 0.3273 | 0.3457 |
| raw_v__construction_mode | 0.7395 | 0.7001 | 0.7300 |
| raw_v__cell__text_only | 0.5869 | 0.5707 | 0.5901 |
| raw_v__cell__vision_grounded | 0.4806 | 0.3620 | 0.4202 |
| text_tfidf__cell | 0.3073 | 0.2844 | 0.2947 |
| raw_v__semantic_target__text_only_editable_to_vision_grounded | 0.4906 | 0.3748 | 0.3873 |
| raw_v__semantic_target__vision_grounded_to_text_only_editable | 0.3971 | 0.3604 | 0.3836 |
| raw_v__edit_operation__text_only_editable_to_vision_grounded | 0.5072 | 0.5045 | 0.5707 |
| raw_v__edit_operation__vision_grounded_to_text_only_editable | 0.6065 | 0.5924 | 0.6336 |

## Factorial structure

- Additive held-out R²: 0.0243
- Saturated 9-cell held-out R²: 0.0326
- Interaction gain: 0.0083

Delete negatives are weaker queries and are not required to contradict the target; 
interpret their delta descriptively rather than as strict retrieval errors.
