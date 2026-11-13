# BGE-VL-base vector visualization

Analyzed 3,401/3,401 edit vectors using `all` sampling.

t-SNE is only a visualization. Use the PCA-space silhouette, adjusted neighbor purity, KMeans ARI/NMI, and projection trustworthiness to decide whether an apparent island is supported quantitatively.

## Quantitative overview

| View | PCA retained variance | Cell silhouette | Cell adjusted neighbor purity | KMeans-9 cell ARI | KMeans-9 cell NMI | t-SNE trustworthiness |
|---|---:|---:|---:|---:|---:|---:|
| raw_v | 0.6446 | -0.0766 | 0.3596 | 0.0546 | 0.2265 | 0.8932 |
| direction_v | 0.6108 | -0.0140 | 0.4048 | 0.1131 | 0.2534 | 0.9524 |

Interpretation:

- `raw_v` contains both direction and step length.
- `direction_v` removes step length and asks whether edit directions alone form the 3×3 structure.
- A t-SNE island without positive original-space silhouette/neighbor purity should be treated as a projection artifact.
- Strong construction-mode structure can confound apparent semantic clusters.
