# BGE-VL-base 3×3 diagnostic

- Samples: 459
- Negatives: 3401
- Embedding dimension: 512
- All probe splits are grouped by pair_id.

## Linear probes

| Probe | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| raw_v__cell | 0.5592 | 0.5430 | 0.5636 |
| raw_v__semantic_target | 0.5634 | 0.5647 | 0.5595 |
| raw_v__edit_operation | 0.7712 | 0.7700 | 0.7836 |
| direction_v__cell | 0.6251 | 0.6066 | 0.6269 |
| norm_v__cell | 0.1702 | 0.1024 | 0.1499 |
| q_positive__cell | 0.1326 | 0.1292 | 0.1431 |
| q_negative__cell | 0.3378 | 0.3184 | 0.3405 |
| raw_v__construction_mode | 0.6854 | 0.6383 | 0.6617 |
| raw_v__cell__text_only | 0.5889 | 0.5724 | 0.5931 |
| raw_v__cell__vision_grounded | 0.4551 | 0.3522 | 0.4230 |
| text_tfidf__cell | 0.3073 | 0.2844 | 0.2947 |
| raw_v__semantic_target__text_only_editable_to_vision_grounded | 0.4297 | 0.3654 | 0.4326 |
| raw_v__semantic_target__vision_grounded_to_text_only_editable | 0.4131 | 0.3826 | 0.3949 |
| raw_v__edit_operation__text_only_editable_to_vision_grounded | 0.7032 | 0.6734 | 0.7071 |
| raw_v__edit_operation__vision_grounded_to_text_only_editable | 0.6721 | 0.6787 | 0.6801 |

## Factorial structure

- Additive held-out R²: 0.0417
- Saturated 9-cell held-out R²: 0.0522
- Interaction gain: 0.0104

Delete negatives are weaker queries and are not required to contradict the target; 
interpret their delta descriptively rather than as strict retrieval errors.
