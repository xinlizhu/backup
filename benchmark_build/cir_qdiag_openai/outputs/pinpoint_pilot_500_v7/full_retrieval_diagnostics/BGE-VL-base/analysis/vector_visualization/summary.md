# BGE-VL-base vector visualization

Analyzed 5,576/5,576 edit vectors using `all` sampling.

t-SNE is only a visualization. Use the PCA-space silhouette, adjusted neighbor purity, KMeans ARI/NMI, and projection trustworthiness to decide whether an apparent island is supported quantitatively.

## Quantitative overview

| View | PCA retained variance | Cell silhouette | Cell adjusted neighbor purity | KMeans-9 cell ARI | KMeans-9 cell NMI | t-SNE trustworthiness |
|---|---:|---:|---:|---:|---:|---:|
| raw_v | 0.5617 | -0.0895 | 0.4068 | 0.0595 | 0.1964 | 0.8721 |
| direction_v | 0.4969 | -0.0080 | 0.4658 | 0.1107 | 0.2073 | 0.9425 |

Interpretation:

- `raw_v` contains both direction and step length.
- `direction_v` removes step length and asks whether edit directions alone form the 3×3 structure.
- A t-SNE island without positive original-space silhouette/neighbor purity should be treated as a projection artifact.
- Strong auxiliary-label structure can confound apparent semantic clusters.
