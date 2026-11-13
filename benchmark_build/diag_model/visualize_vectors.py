#!/usr/bin/env python3
"""Visualize and quantify structure in CIR-QDiag edit vectors."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cir-qdiag-matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import TSNE, trustworthiness  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors  # noqa: E402

from common import (
    CELL_LABELS,
    CONSTRUCTION_MODES,
    EDIT_OPERATIONS,
    SEMANTIC_TARGETS,
    read_jsonl,
    write_json_atomic,
)
from probe_grid import label_arrays, load_features, make_feature_views


VIEW_NAMES = ("raw_v", "direction_v")
LABEL_ORDERS = {
    "cell": CELL_LABELS,
    "semantic_target": SEMANTIC_TARGETS,
    "edit_operation": EDIT_OPERATIONS,
    "construction_mode": CONSTRUCTION_MODES,
    "review_scope": ("fully_reviewed", "partially_reviewed"),
}
LABEL_TITLES = {
    "cell": "3×3 cell",
    "semantic_target": "Semantic target",
    "edit_operation": "Edit operation",
    "construction_mode": "Construction mode",
    "review_scope": "Positive-image review scope",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create PCA/t-SNE plots and unsupervised structure metrics for "
            "v=q_positive-q_negative."
        )
    )
    parser.add_argument(
        "--features-dir",
        required=True,
        type=Path,
        help="Directory produced by extract_embeddings.py.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--views",
        default=",".join(VIEW_NAMES),
        help=f"Comma-separated subset of {','.join(VIEW_NAMES)}.",
    )
    parser.add_argument(
        "--pca-dim",
        type=int,
        default=50,
        help="PCA dimension used before t-SNE and for quantitative metrics.",
    )
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-iterations", type=int, default=1500)
    parser.add_argument(
        "--neighbors",
        type=int,
        default=15,
        help="Neighborhood size for trustworthiness and label-purity metrics.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=5000,
        help="Maximum plotted/analyzed rows; larger inputs are cell-balanced sampled.",
    )
    parser.add_argument(
        "--silhouette-sample-size",
        type=int,
        default=2000,
        help="Maximum rows used by each silhouette calculation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[str, ...]:
    views = tuple(value.strip() for value in args.views.split(",") if value.strip())
    unknown = sorted(set(views) - set(VIEW_NAMES))
    if unknown:
        raise ValueError(f"Unknown views: {unknown}; supported={VIEW_NAMES}")
    if not views:
        raise ValueError("--views selected no feature views")
    if args.pca_dim < 2:
        raise ValueError("--pca-dim must be at least 2")
    if args.perplexity <= 0:
        raise ValueError("--perplexity must be positive")
    if args.tsne_iterations < 250:
        raise ValueError("--tsne-iterations must be at least 250")
    if args.neighbors < 1:
        raise ValueError("--neighbors must be positive")
    if args.max_points < len(CELL_LABELS):
        raise ValueError(f"--max-points must be at least {len(CELL_LABELS)}")
    if args.silhouette_sample_size < 2:
        raise ValueError("--silhouette-sample-size must be at least 2")
    return views


def choose_rows(
    cell_labels: np.ndarray,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, str]:
    """Return all rows or a reproducible approximately cell-balanced sample."""

    row_count = len(cell_labels)
    if row_count <= max_points:
        return np.arange(row_count, dtype=np.int64), "all"

    rng = np.random.default_rng(seed)
    cell_to_indices = {
        cell: np.flatnonzero(cell_labels == cell)
        for cell in CELL_LABELS
        if np.any(cell_labels == cell)
    }
    selected: list[int] = []
    remaining = max_points
    active = list(cell_to_indices)
    while active and remaining:
        quota = max(1, remaining // len(active))
        next_active: list[str] = []
        for cell in active:
            available = cell_to_indices[cell]
            take = min(quota, len(available))
            if take:
                picked = rng.choice(available, size=take, replace=False)
                selected.extend(int(value) for value in picked)
                cell_to_indices[cell] = np.setdiff1d(
                    available, picked, assume_unique=False
                )
                remaining -= take
            if len(cell_to_indices[cell]) and remaining:
                next_active.append(cell)
            if not remaining:
                break
        active = next_active
    return np.asarray(sorted(selected), dtype=np.int64), "cell_balanced"


def safe_silhouette(
    matrix: np.ndarray,
    labels: np.ndarray,
    sample_size: int,
    seed: int,
) -> float:
    counts = Counter(labels.tolist())
    if len(counts) < 2 or min(counts.values()) < 2:
        return float("nan")
    return float(
        silhouette_score(
            matrix,
            labels,
            metric="euclidean",
            sample_size=min(sample_size, len(labels)),
            random_state=seed,
        )
    )


def neighbor_label_purity(
    matrix: np.ndarray,
    labels: np.ndarray,
    neighbors: int,
) -> dict[str, float]:
    if len(matrix) < 2:
        return {
            "neighbor_purity": float("nan"),
            "chance_purity": float("nan"),
            "adjusted_neighbor_purity": float("nan"),
        }
    k = min(neighbors, len(matrix) - 1)
    model = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    indices = model.fit(matrix).kneighbors(matrix, return_distance=False)
    neighbor_indices = indices[:, 1:]
    purity = float(np.mean(labels[neighbor_indices] == labels[:, None]))
    frequencies = np.asarray(list(Counter(labels.tolist()).values()), dtype=np.float64)
    frequencies /= frequencies.sum()
    chance = float(np.sum(frequencies**2))
    adjusted = (purity - chance) / (1.0 - chance) if chance < 1.0 else float("nan")
    return {
        "neighbor_purity": purity,
        "chance_purity": chance,
        "adjusted_neighbor_purity": float(adjusted),
    }


def label_space_metrics(
    matrix: np.ndarray,
    labels: dict[str, np.ndarray],
    neighbors: int,
    silhouette_sample_size: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    report: dict[str, dict[str, float]] = {}
    for label_name, values in labels.items():
        metrics = neighbor_label_purity(matrix, values, neighbors)
        metrics["silhouette"] = safe_silhouette(
            matrix,
            values,
            silhouette_sample_size,
            seed,
        )
        report[label_name] = metrics
    return report


def kmeans_metrics(
    matrix: np.ndarray,
    labels: dict[str, np.ndarray],
    seed: int,
    label_names: Sequence[str] | None = None,
) -> dict[str, dict[str, float | int]]:
    report: dict[str, dict[str, float | int]] = {}
    if label_names is None:
        label_names = tuple(name for name in labels if name not in {"group", "pair_id"})
    for label_name in label_names:
        values = labels[label_name]
        cluster_count = len(set(values.tolist()))
        if cluster_count < 2 or len(matrix) <= cluster_count:
            continue
        predictions = KMeans(
            n_clusters=cluster_count,
            n_init=20,
            random_state=seed,
        ).fit_predict(matrix)
        report[label_name] = {
            "cluster_count": cluster_count,
            "adjusted_rand_index": float(adjusted_rand_score(values, predictions)),
            "normalized_mutual_information": float(
                normalized_mutual_info_score(values, predictions)
            ),
            "homogeneity": float(homogeneity_score(values, predictions)),
            "completeness": float(completeness_score(values, predictions)),
        }
    return report


def project_view(
    matrix: np.ndarray,
    pca_dim: int,
    perplexity: float,
    iterations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | int]]:
    analysis_dimension = min(pca_dim, matrix.shape[1], matrix.shape[0] - 1)
    if analysis_dimension < 2:
        raise ValueError("At least three rows and two feature dimensions are required")
    pca = PCA(
        n_components=analysis_dimension,
        svd_solver="randomized",
        random_state=seed,
    )
    analysis_matrix = pca.fit_transform(matrix).astype(np.float32, copy=False)
    pca_2d = analysis_matrix[:, :2]
    effective_perplexity = min(perplexity, max(1.0, (len(matrix) - 1) / 3.0))
    tsne_2d = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        early_exaggeration=12.0,
        learning_rate="auto",
        max_iter=iterations,
        init="pca",
        metric="euclidean",
        random_state=seed,
        method="barnes_hut",
    ).fit_transform(analysis_matrix)
    metadata: dict[str, float | int] = {
        "analysis_dimension": int(analysis_dimension),
        "pca_explained_variance_ratio_sum": float(
            np.sum(pca.explained_variance_ratio_)
        ),
        "pca_2d_explained_variance_ratio_sum": float(
            np.sum(pca.explained_variance_ratio_[:2])
        ),
        "effective_perplexity": float(effective_perplexity),
    }
    return analysis_matrix, pca_2d, tsne_2d, metadata


def plot_projection(
    coordinates: np.ndarray,
    labels: dict[str, np.ndarray],
    method: str,
    view_name: str,
    output_path: Path,
    dpi: int,
    label_names: Sequence[str] = (
        "cell",
        "semantic_target",
        "edit_operation",
        "construction_mode",
    ),
) -> None:
    if len(label_names) != 4:
        raise ValueError("plot_projection requires exactly four label panels")
    figure, axes = plt.subplots(2, 2, figsize=(16, 13), constrained_layout=True)
    colormaps = {
        "cell": "tab20",
        "semantic_target": "Set1",
        "edit_operation": "Dark2",
        "construction_mode": "Set2",
        "review_scope": "Set2",
    }
    for axis, label_name in zip(axes.flat, label_names):
        values = labels[label_name]
        configured_order = LABEL_ORDERS.get(
            label_name,
            tuple(sorted(set(values.tolist()))),
        )
        order = [value for value in configured_order if np.any(values == value)]
        palette = plt.get_cmap(colormaps.get(label_name, "tab20"))(
            np.linspace(0.0, 0.85, max(1, len(order)))
        )
        for color, value in zip(palette, order):
            mask = values == value
            axis.scatter(
                coordinates[mask, 0],
                coordinates[mask, 1],
                s=10,
                alpha=0.55,
                linewidths=0,
                color=color,
                label=value.replace("|", " × "),
                rasterized=True,
            )
            if label_name == "cell":
                center = np.median(coordinates[mask], axis=0)
                axis.annotate(
                    value.replace("|", "\n"),
                    center,
                    fontsize=7,
                    ha="center",
                    va="center",
                    bbox={
                        "boxstyle": "round,pad=0.2",
                        "facecolor": "white",
                        "edgecolor": color,
                        "alpha": 0.78,
                    },
                )
        axis.set_title(LABEL_TITLES.get(label_name, label_name.replace("_", " ").title()))
        axis.set_xticks([])
        axis.set_yticks([])
        axis.grid(False)
        legend_columns = 3 if label_name == "cell" else 1
        axis.legend(
            loc="best",
            fontsize=7 if label_name == "cell" else 8,
            frameon=True,
            ncol=legend_columns,
            markerscale=1.8,
        )
    figure.suptitle(
        f"{method.upper()} projection of {view_name} (v = q+ − q−)",
        fontsize=16,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def write_rows_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def flatten_quality_metrics(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for view_name, view in report["views"].items():
        for space_name in ("analysis_space", "pca_2d", "tsne_2d"):
            for label_name, metrics in view[space_name]["label_metrics"].items():
                for metric_name, value in metrics.items():
                    rows.append(
                        {
                            "view": view_name,
                            "space": space_name,
                            "label": label_name,
                            "metric": metric_name,
                            "value": value,
                        }
                    )
        for label_name, metrics in view["analysis_space"]["kmeans"].items():
            for metric_name, value in metrics.items():
                rows.append(
                    {
                        "view": view_name,
                        "space": "analysis_space_kmeans",
                        "label": label_name,
                        "metric": metric_name,
                        "value": value,
                    }
                )
        for space_name in ("pca_2d", "tsne_2d"):
            rows.append(
                {
                    "view": view_name,
                    "space": space_name,
                    "label": "all",
                    "metric": "trustworthiness",
                    "value": view[space_name]["trustworthiness"],
                }
            )
    return rows


def make_summary(report: dict[str, Any]) -> str:
    lines = [
        f"# {report['model_name']} vector visualization",
        "",
        (
            f"Analyzed {report['analyzed_row_count']:,}/{report['source_row_count']:,} "
            f"edit vectors using `{report['sampling_strategy']}` sampling."
        ),
        "",
        "t-SNE is only a visualization. Use the PCA-space silhouette, adjusted "
        "neighbor purity, KMeans ARI/NMI, and projection trustworthiness to decide "
        "whether an apparent island is supported quantitatively.",
        "",
        "## Quantitative overview",
        "",
        "| View | PCA retained variance | Cell silhouette | Cell adjusted neighbor purity | "
        "KMeans-9 cell ARI | KMeans-9 cell NMI | t-SNE trustworthiness |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for view_name, view in report["views"].items():
        cell = view["analysis_space"]["label_metrics"]["cell"]
        kmeans = view["analysis_space"]["kmeans"]["cell"]
        lines.append(
            "| "
            + " | ".join(
                [
                    view_name,
                    f"{view['projection_metadata']['pca_explained_variance_ratio_sum']:.4f}",
                    f"{cell['silhouette']:.4f}",
                    f"{cell['adjusted_neighbor_purity']:.4f}",
                    f"{kmeans['adjusted_rand_index']:.4f}",
                    f"{kmeans['normalized_mutual_information']:.4f}",
                    f"{view['tsne_2d']['trustworthiness']:.4f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- `raw_v` contains both direction and step length.",
            "- `direction_v` removes step length and asks whether edit directions alone "
            "form the 3×3 structure.",
            "- A t-SNE island without positive original-space silhouette/neighbor purity "
            "should be treated as a projection artifact.",
            "- Strong auxiliary-label structure can confound apparent semantic clusters.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    views = validate_args(args)
    arrays, rows, extraction_metadata = load_features(args.features_dir)
    features, _ = make_feature_views(arrays)
    labels_all = label_arrays(rows)
    selected, sampling_strategy = choose_rows(
        labels_all["cell"],
        args.max_points,
        args.seed,
    )
    labels = {name: values[selected] for name, values in labels_all.items()}
    selected_rows = [rows[int(index)] for index in selected]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "model_name": extraction_metadata.get("model_name", args.features_dir.name),
        "features_dir": str(args.features_dir.resolve()),
        "source_row_count": len(rows),
        "analyzed_row_count": len(selected),
        "sampling_strategy": sampling_strategy,
        "seed": args.seed,
        "neighbors": args.neighbors,
        "requested_pca_dimension": args.pca_dim,
        "requested_perplexity": args.perplexity,
        "tsne_iterations": args.tsne_iterations,
        "label_counts": {
            name: dict(sorted(Counter(values.tolist()).items()))
            for name, values in labels.items()
        },
        "views": {},
    }
    coordinate_columns: dict[str, np.ndarray] = {}

    for view_name in views:
        matrix = np.asarray(features[view_name][selected], dtype=np.float32)
        print(
            f"[projection] model={report['model_name']} view={view_name} "
            f"rows={len(matrix)} dimension={matrix.shape[1]}",
            flush=True,
        )
        analysis_matrix, pca_2d, tsne_2d, projection_metadata = project_view(
            matrix,
            args.pca_dim,
            args.perplexity,
            args.tsne_iterations,
            args.seed,
        )
        effective_neighbors = min(
            args.neighbors,
            max(1, (len(matrix) - 1) // 2),
        )
        analysis_metrics = label_space_metrics(
            analysis_matrix,
            labels,
            effective_neighbors,
            args.silhouette_sample_size,
            args.seed,
        )
        view_report = {
            "projection_metadata": projection_metadata,
            "analysis_space": {
                "label_metrics": analysis_metrics,
                "kmeans": kmeans_metrics(analysis_matrix, labels, args.seed),
            },
            "pca_2d": {
                "trustworthiness": float(
                    trustworthiness(
                        analysis_matrix,
                        pca_2d,
                        n_neighbors=effective_neighbors,
                        metric="euclidean",
                    )
                ),
                "label_metrics": label_space_metrics(
                    pca_2d,
                    labels,
                    effective_neighbors,
                    args.silhouette_sample_size,
                    args.seed,
                ),
            },
            "tsne_2d": {
                "trustworthiness": float(
                    trustworthiness(
                        analysis_matrix,
                        tsne_2d,
                        n_neighbors=effective_neighbors,
                        metric="euclidean",
                    )
                ),
                "label_metrics": label_space_metrics(
                    tsne_2d,
                    labels,
                    effective_neighbors,
                    args.silhouette_sample_size,
                    args.seed,
                ),
            },
        }
        report["views"][view_name] = view_report
        coordinate_columns[f"{view_name}_pca_x"] = pca_2d[:, 0]
        coordinate_columns[f"{view_name}_pca_y"] = pca_2d[:, 1]
        coordinate_columns[f"{view_name}_tsne_x"] = tsne_2d[:, 0]
        coordinate_columns[f"{view_name}_tsne_y"] = tsne_2d[:, 1]
        plot_projection(
            pca_2d,
            labels,
            "PCA",
            view_name,
            args.output_dir / f"pca_{view_name}.png",
            args.dpi,
        )
        plot_projection(
            tsne_2d,
            labels,
            "t-SNE",
            view_name,
            args.output_dir / f"tsne_{view_name}.png",
            args.dpi,
        )

    coordinate_rows: list[dict[str, Any]] = []
    for local_index, (source_index, row) in enumerate(zip(selected, selected_rows)):
        output_row: dict[str, Any] = {
            "source_index": int(source_index),
            "negative_index": int(row["negative_index"]),
            "sample_index": int(row["sample_index"]),
            "sample_id": row.get("sample_id", ""),
            "pair_id": row["pair_id"],
            "negative_id": row["negative_id"],
            "semantic_target": row["semantic_target"],
            "edit_operation": row["edit_operation"],
            "cell": f"{row['semantic_target']}|{row['edit_operation']}",
            "construction_mode": row["construction_mode"],
        }
        for name, values in coordinate_columns.items():
            output_row[name] = float(values[local_index])
        coordinate_rows.append(output_row)

    write_rows_csv(args.output_dir / "coordinates.csv", coordinate_rows)
    write_rows_csv(
        args.output_dir / "quality_metrics.csv",
        flatten_quality_metrics(report),
    )
    write_json_atomic(args.output_dir / "report.json", report)
    (args.output_dir / "summary.md").write_text(
        make_summary(report),
        encoding="utf-8",
    )
    print(f"[done] wrote vector visualization to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
