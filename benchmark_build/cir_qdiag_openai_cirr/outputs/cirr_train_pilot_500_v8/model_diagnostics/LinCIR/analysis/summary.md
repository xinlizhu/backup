# LinCIR 3×3 diagnostic

- Samples: 500
- Negatives: 2446
- Embedding dimension: 768
- All probe splits are grouped by pair_id.
- Reference ablation f_query(R, empty) is available in reference_ablation_by_cell.csv.

## Linear probes

| Probe | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| raw_v__cell | 0.5454 | 0.5276 | 0.5628 |
| raw_v__semantic_target | 0.5859 | 0.5877 | 0.5876 |
| raw_v__edit_operation | 0.7375 | 0.7422 | 0.7515 |
| direction_v__cell | 0.6092 | 0.5990 | 0.6188 |
| norm_v__cell | 0.2003 | 0.1388 | 0.1746 |
| q_positive__cell | 0.1206 | 0.1195 | 0.1340 |
| q_negative__cell | 0.3062 | 0.2833 | 0.3165 |
| text_tfidf__cell | 0.2715 | 0.2569 | 0.2733 |

## Factorial structure

- Additive held-out R²: 0.0289
- Saturated 9-cell held-out R²: 0.0386
- Interaction gain: 0.0097

Delete negatives are weaker queries and are not required to contradict the target; 
interpret their delta descriptively rather than as strict retrieval errors.
