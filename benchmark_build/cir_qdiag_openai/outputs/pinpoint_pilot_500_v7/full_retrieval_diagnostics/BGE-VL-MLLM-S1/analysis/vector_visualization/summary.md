# BGE-VL-MLLM-S1 vector visualization

Analyzed 5,576/5,576 edit vectors using `all` sampling.

t-SNE is only a visualization. Use the PCA-space silhouette, adjusted neighbor purity, KMeans ARI/NMI, and projection trustworthiness to decide whether an apparent island is supported quantitatively.

## Quantitative overview

| View | PCA retained variance | Cell silhouette | Cell adjusted neighbor purity | KMeans-9 cell ARI | KMeans-9 cell NMI | t-SNE trustworthiness |
|---|---:|---:|---:|---:|---:|---:|
| raw_v | 0.4685 | -0.1401 | 0.4156 | 0.0908 | 0.2422 | 0.8818 |
| direction_v | 0.3939 | -0.0469 | 0.4550 | 0.1419 | 0.2530 | 0.9495 |

Interpretation:

- `raw_v` contains both direction and step length.
- `direction_v` removes step length and asks whether edit directions alone form the 3×3 structure.
- A t-SNE island without positive original-space silhouette/neighbor purity should be treated as a projection artifact.
- Strong auxiliary-label structure can confound apparent semantic clusters.
