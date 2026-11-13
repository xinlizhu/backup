#!/usr/bin/env python3
"""PCA/t-SNE diagnostics for q+ - q- vectors cached by PinPoint V7 retrieval runs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.manifold import trustworthiness

from common import CELL_LABELS, EDIT_OPERATIONS, SEMANTIC_TARGETS, read_jsonl, write_json_atomic
from visualize_vectors import (
    VIEW_NAMES,
    choose_rows,
    flatten_quality_metrics,
    kmeans_metrics,
    label_space_metrics,
    make_summary,
    plot_projection,
    project_view,
    write_rows_csv,
)


PLOT_LABEL_NAMES = (
    "cell",
    "semantic_target",
    "edit_operation",
    "review_scope",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize PinPoint V7 q_positive-q_negative vectors from query caches."
    )
    parser.add_argument("--input", required=True, type=Path, help="PinPoint V7 final.jsonl")
    parser.add_argument(
        "--model-dir",
        required=True,
        type=Path,
        help="One model directory produced by pinpoint_full_diagnostics.py.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--views", default=",".join(VIEW_NAMES))
    parser.add_argument("--pca-dim", type=int, default=50)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-iterations", type=int, default=1500)
    parser.add_argument("--neighbors", type=int, default=15)
    parser.add_argument("--max-points", type=int, default=6000)
    parser.add_argument("--silhouette-sample-size", type=int, default=2000)
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
    return views


def load_query_cache(model_dir: Path) -> tuple[np.ndarray, dict[str, int], dict[str, Any]]:
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing PinPoint model metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    fingerprint = str(metadata.get("query_fingerprint", ""))
    if not fingerprint:
        raise ValueError(f"{metadata_path} has no query_fingerprint")
    cache_dir = model_dir / "cache" / "queries" / fingerprint[:16]
    embeddings_path = cache_dir / "query_embeddings.npy"
    keys_path = cache_dir / "query_keys.npy"
    cache_metadata_path = cache_dir / "query_metadata.json"
    for path in (embeddings_path, keys_path, cache_metadata_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing PinPoint query cache: {path}")
    cache_metadata = json.loads(cache_metadata_path.read_text(encoding="utf-8"))
    if cache_metadata.get("fingerprint") != fingerprint:
        raise ValueError("PinPoint query cache fingerprint does not match model metadata")
    embeddings = np.load(embeddings_path, mmap_mode="r")
    keys = np.load(keys_path, allow_pickle=False).astype(str).tolist()
    if embeddings.ndim != 2 or len(keys) != len(embeddings):
        raise ValueError("PinPoint query keys and embeddings have inconsistent shapes")
    key_index = {value: index for index, value in enumerate(keys)}
    if len(key_index) != len(keys):
        raise ValueError("PinPoint query cache contains duplicate query keys")
    return embeddings, key_index, metadata


def load_pinpoint_vectors(
    input_path: Path,
    model_dir: Path,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    embeddings, key_index, metadata = load_query_cache(model_dir)
    rows: list[dict[str, Any]] = []
    positive_indices: list[int] = []
    negative_indices: list[int] = []
    for group in read_jsonl(input_path):
        group_id = str(group["group_id"])
        instructions = {
            str(value["instruction_id"]): value for value in group["instructions"]
        }
        positive_count = len(group.get("positive_images", []))
        validated_count = len(
            group.get("metadata", {}).get("validated_positive_image_ids", [])
        )
        review_scope = (
            "fully_reviewed"
            if positive_count == validated_count
            else "partially_reviewed"
        )
        for negative in group.get("negatives", []):
            instruction_id = str(negative["instruction_id"])
            negative_id = str(negative["negative_id"])
            positive_key = f"positive::{instruction_id}"
            negative_key = f"negative::{negative_id}"
            missing = [
                value
                for value in (positive_key, negative_key)
                if value not in key_index
            ]
            if missing:
                raise KeyError(
                    f"{group_id}/{negative_id} is missing cached queries: {missing}"
                )
            instruction = instructions[instruction_id]
            positive_indices.append(key_index[positive_key])
            negative_indices.append(key_index[negative_key])
            rows.append(
                {
                    "negative_index": len(rows),
                    "sample_index": key_index[positive_key],
                    "sample_id": instruction_id,
                    "pair_id": group_id,
                    "group_id": group_id,
                    "instruction_id": instruction_id,
                    "negative_id": negative_id,
                    "positive_text": str(instruction["text"]),
                    "negative_text": str(negative["text"]),
                    "semantic_target": str(negative["semantic_target"]),
                    "edit_operation": str(negative["edit_operation"]),
                    "query_category": str(instruction.get("query_category", "")),
                    "reference_count": len(group.get("references", [])),
                    "positive_image_count": positive_count,
                    "validated_positive_image_count": validated_count,
                    "review_scope": review_scope,
                }
            )
    if not rows:
        raise ValueError(f"{input_path} contains no accepted PinPoint negatives")
    positive = np.asarray(embeddings[np.asarray(positive_indices)], dtype=np.float32)
    negative = np.asarray(embeddings[np.asarray(negative_indices)], dtype=np.float32)
    raw = positive - negative
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    direction = np.divide(
        raw,
        norms,
        out=np.zeros_like(raw),
        where=norms > 1e-12,
    )
    if not np.isfinite(raw).all() or not np.isfinite(direction).all():
        raise ValueError("PinPoint edit vectors contain non-finite values")
    return {"raw_v": raw, "direction_v": direction}, rows, metadata


def make_labels(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    semantic = np.asarray([row["semantic_target"] for row in rows], dtype=object)
    operation = np.asarray([row["edit_operation"] for row in rows], dtype=object)
    return {
        "cell": np.asarray(
            [
                f"{semantic_value}|{operation_value}"
                for semantic_value, operation_value in zip(semantic, operation)
            ],
            dtype=object,
        ),
        "semantic_target": semantic,
        "edit_operation": operation,
        "review_scope": np.asarray([row["review_scope"] for row in rows], dtype=object),
        "group": np.asarray([row["group_id"] for row in rows], dtype=object),
    }


def main() -> None:
    args = parse_args()
    views = validate_args(args)
    input_path = args.input.resolve()
    model_dir = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()
    features, rows, model_metadata = load_pinpoint_vectors(input_path, model_dir)
    labels_all = make_labels(rows)
    selected, sampling_strategy = choose_rows(
        labels_all["cell"],
        args.max_points,
        args.seed,
    )
    labels = {name: values[selected] for name, values in labels_all.items()}
    selected_rows = [rows[int(index)] for index in selected]
    output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "source_kind": "pinpoint_v7_query_cache",
        "model_name": model_metadata.get("model_name", model_dir.name),
        "input_path": str(input_path),
        "model_dir": str(model_dir),
        "query_fingerprint": model_metadata.get("query_fingerprint"),
        "source_row_count": len(rows),
        "analyzed_row_count": len(selected),
        "sampling_strategy": sampling_strategy,
        "seed": args.seed,
        "neighbors": args.neighbors,
        "requested_pca_dimension": args.pca_dim,
        "requested_perplexity": args.perplexity,
        "tsne_iterations": args.tsne_iterations,
        "plot_label_names": list(PLOT_LABEL_NAMES),
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
            f"[projection] dataset=PinPoint-V7 model={report['model_name']} "
            f"view={view_name} rows={len(matrix)} dimension={matrix.shape[1]}",
            flush=True,
        )
        analysis_matrix, pca_2d, tsne_2d, projection_metadata = project_view(
            matrix,
            args.pca_dim,
            args.perplexity,
            args.tsne_iterations,
            args.seed,
        )
        effective_neighbors = min(args.neighbors, max(1, (len(matrix) - 1) // 2))
        view_report = {
            "projection_metadata": projection_metadata,
            "analysis_space": {
                "label_metrics": label_space_metrics(
                    analysis_matrix,
                    labels,
                    effective_neighbors,
                    args.silhouette_sample_size,
                    args.seed,
                ),
                "kmeans": kmeans_metrics(
                    analysis_matrix,
                    labels,
                    args.seed,
                    PLOT_LABEL_NAMES,
                ),
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
            output_dir / f"pca_{view_name}.png",
            args.dpi,
            PLOT_LABEL_NAMES,
        )
        plot_projection(
            tsne_2d,
            labels,
            "t-SNE",
            view_name,
            output_dir / f"tsne_{view_name}.png",
            args.dpi,
            PLOT_LABEL_NAMES,
        )

    coordinate_rows: list[dict[str, Any]] = []
    for local_index, (source_index, row) in enumerate(zip(selected, selected_rows)):
        output_row = {"source_index": int(source_index), **row}
        for name, values in coordinate_columns.items():
            output_row[name] = float(values[local_index])
        coordinate_rows.append(output_row)
    write_rows_csv(output_dir / "coordinates.csv", coordinate_rows)
    write_rows_csv(
        output_dir / "quality_metrics.csv",
        flatten_quality_metrics(report),
    )
    write_json_atomic(output_dir / "report.json", report)
    (output_dir / "summary.md").write_text(make_summary(report), encoding="utf-8")
    print(f"[done] wrote PinPoint V7 vector visualization to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
