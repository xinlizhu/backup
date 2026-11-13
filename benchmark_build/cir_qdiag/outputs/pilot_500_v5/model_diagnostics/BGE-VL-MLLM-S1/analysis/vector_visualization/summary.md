# BGE-VL-MLLM-S1 vector visualization

Analyzed 3,401/3,401 edit vectors using `all` sampling.

t-SNE is only a visualization. Use the PCA-space silhouette, adjusted neighbor purity, KMeans ARI/NMI, and projection trustworthiness to decide whether an apparent island is supported quantitatively.

## Quantitative overview

| View | PCA retained variance | Cell silhouette | Cell adjusted neighbor purity | KMeans-9 cell ARI | KMeans-9 cell NMI | t-SNE trustworthiness |
|---|---:|---:|---:|---:|---:|---:|
| raw_v | 0.5164 | -0.0641 | 0.3561 | 0.0434 | 0.2141 | 0.9014 |
| direction_v | 0.4797 | -0.0461 | 0.3995 | 0.0867 | 0.2294 | 0.9542 |

Interpretation:

- `raw_v` contains both direction and step length.
- `direction_v` removes step length and asks whether edit directions alone form the 3×3 structure.
- A t-SNE island without positive original-space silhouette/neighbor purity should be treated as a projection artifact.
- Strong construction-mode structure can confound apparent semantic clusters.
