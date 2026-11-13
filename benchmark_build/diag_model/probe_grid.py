#!/usr/bin/env python3
"""Offline 3x3 diagnostics for cached CIR-QDiag model embeddings."""

from __future__ import annotations

import argparse
import csv
import json
import os
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import (
    CELL_LABELS,
    CONSTRUCTION_MODES,
    EDIT_OPERATIONS,
    SEMANTIC_TARGETS,
    read_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze delta and train leakage-safe linear probes on cached embeddings."
    )
    parser.add_argument(
        "--features-dir",
        required=True,
        type=Path,
        help="Directory produced by extract_embeddings.py.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=2)
    parser.add_argument("--c-grid", default="0.1,1.0")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iterations", type=int, default=3000)
    parser.add_argument(
        "--descriptive-only",
        action="store_true",
        help=(
            "Only refresh delta_by_cell and reference-ablation statistics; "
            "do not retrain probes or factor models."
        ),
    )
    parser.add_argument(
        "--factor-ridge",
        type=float,
        default=1e-3,
        help="Small ridge penalty for the multivariate additive/saturated factor models.",
    )
    return parser.parse_args()


def load_features(features_dir: Path) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    feature_path = features_dir / "features.npz"
    rows_path = features_dir / "rows.jsonl"
    metadata_path = features_dir / "metadata.json"
    for path in (feature_path, rows_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required extraction output is missing: {path}")

    with np.load(feature_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    rows = list(read_jsonl(rows_path))
    rows.sort(key=lambda item: int(item["negative_index"]))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    required_arrays = {"q_positive", "q_negative", "target", "sample_index"}
    missing = required_arrays - arrays.keys()
    if missing:
        raise ValueError(f"features.npz is missing arrays: {sorted(missing)}")
    negative_count = arrays["q_negative"].shape[0]
    if len(rows) != negative_count or arrays["sample_index"].shape != (negative_count,):
        raise ValueError(
            "rows.jsonl, q_negative, and sample_index have inconsistent row counts"
        )
    if any(int(row["negative_index"]) != index for index, row in enumerate(rows)):
        raise ValueError("rows.jsonl negative_index values must be contiguous and ordered")
    if arrays["sample_index"].min(initial=0) < 0 or arrays["sample_index"].max(initial=0) >= arrays["q_positive"].shape[0]:
        raise ValueError("sample_index contains an out-of-range value")
    if "q_reference" in arrays and arrays["q_reference"].shape != arrays["q_positive"].shape:
        raise ValueError("q_reference and q_positive must have the same shape")
    return arrays, rows, metadata


def label_arrays(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    semantic = np.asarray([row["semantic_target"] for row in rows], dtype=object)
    operation = np.asarray([row["edit_operation"] for row in rows], dtype=object)
    return {
        "cell": np.asarray(
            [f"{target}|{edit}" for target, edit in zip(semantic, operation)],
            dtype=object,
        ),
        "semantic_target": semantic,
        "edit_operation": operation,
        "construction_mode": np.asarray(
            [row["construction_mode"] for row in rows], dtype=object
        ),
        "group": np.asarray([str(row["pair_id"]) for row in rows], dtype=object),
    }


def sphere_step_metrics(
    start: np.ndarray,
    end: np.ndarray,
    target: np.ndarray,
) -> dict[str, np.ndarray]:
    """Describe a finite normalized-embedding step and its target alignment."""

    start_score = np.einsum("nd,nd->n", start, target)
    end_score = np.einsum("nd,nd->n", end, target)
    gain = end_score - start_score
    chord = end - start
    chord_norm = np.linalg.norm(chord, axis=1)
    chord_target_cosine = np.divide(
        gain,
        chord_norm,
        out=np.zeros_like(gain),
        where=chord_norm > 1e-12,
    )

    endpoint_cosine = np.clip(np.einsum("nd,nd->n", start, end), -1.0, 1.0)
    angle_degrees = np.degrees(np.arccos(endpoint_cosine))
    tangent = end - endpoint_cosine[:, None] * start
    tangent_norm = np.linalg.norm(tangent, axis=1)
    target_gradient = target - start_score[:, None] * start
    gradient_norm = np.linalg.norm(target_gradient, axis=1)
    alignment_denominator = tangent_norm * gradient_norm
    local_target_alignment = np.divide(
        np.einsum("nd,nd->n", tangent, target_gradient),
        alignment_denominator,
        out=np.zeros_like(gain),
        where=alignment_denominator > 1e-12,
    )
    return {
        "start_score": start_score,
        "end_score": end_score,
        "gain": gain,
        "chord": chord,
        "chord_norm": chord_norm,
        "chord_target_cosine": chord_target_cosine,
        "endpoint_cosine": endpoint_cosine,
        "angle_degrees": angle_degrees,
        "local_target_alignment": local_target_alignment,
    }


def make_feature_views(arrays: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    sample_index = arrays["sample_index"].astype(np.int64, copy=False)
    q_positive = arrays["q_positive"].astype(np.float32, copy=False)[sample_index]
    q_negative = arrays["q_negative"].astype(np.float32, copy=False)
    target = arrays["target"].astype(np.float32, copy=False)[sample_index]
    query_step = sphere_step_metrics(q_negative, q_positive, target)
    vector = query_step["chord"]
    vector_norm = query_step["chord_norm"]
    direction = np.divide(
        vector,
        vector_norm[:, None],
        out=np.zeros_like(vector),
        where=vector_norm[:, None] > 1e-12,
    )
    positive_score = query_step["end_score"]
    negative_score = query_step["start_score"]
    delta = query_step["gain"]
    delta_from_vector = np.einsum("nd,nd->n", vector, target)
    identity_error = np.abs(delta - delta_from_vector)
    if float(identity_error.max(initial=0.0)) > 1e-5:
        raise AssertionError("Cached features fail delta = dot(q+ - q-, t)")
    vector_target_cosine = query_step["chord_target_cosine"]
    features = {
        "raw_v": vector,
        "direction_v": direction,
        "norm_v": vector_norm[:, None],
        "q_positive": q_positive,
        "q_negative": q_negative,
    }
    scores = {
        "positive_score": positive_score,
        "negative_score": negative_score,
        "delta": delta,
        "vector_norm": vector_norm,
        "vector_target_cosine": vector_target_cosine,
        "query_angle_degrees": query_step["angle_degrees"],
        "local_target_alignment": query_step["local_target_alignment"],
        "identity_error": identity_error,
    }
    if "q_reference" in arrays:
        q_reference = arrays["q_reference"].astype(np.float32, copy=False)[sample_index]
        positive_step = sphere_step_metrics(q_reference, q_positive, target)
        negative_step = sphere_step_metrics(q_reference, q_negative, target)
        positive_shift = positive_step["chord"]
        negative_shift = negative_step["chord"]
        shift_norm_product = positive_step["chord_norm"] * negative_step["chord_norm"]
        shift_direction_cosine = np.divide(
            np.einsum("nd,nd->n", positive_shift, negative_shift),
            shift_norm_product,
            out=np.zeros_like(shift_norm_product),
            where=shift_norm_product > 1e-12,
        )
        contribution_identity_error = np.abs(
            delta - (positive_step["gain"] - negative_step["gain"])
        )
        if float(contribution_identity_error.max(initial=0.0)) > 1e-5:
            raise AssertionError("Cached features fail delta = G+ - G-")
        features["q_reference"] = q_reference
        scores.update(
            {
                "reference_score": positive_step["start_score"],
                "positive_gain_from_reference": positive_step["gain"],
                "negative_gain_from_reference": negative_step["gain"],
                "positive_reference_distance": positive_step["chord_norm"],
                "negative_reference_distance": negative_step["chord_norm"],
                "positive_reference_angle_degrees": positive_step["angle_degrees"],
                "negative_reference_angle_degrees": negative_step["angle_degrees"],
                "positive_reference_target_cosine": positive_step["chord_target_cosine"],
                "negative_reference_target_cosine": negative_step["chord_target_cosine"],
                "positive_reference_local_alignment": positive_step[
                    "local_target_alignment"
                ],
                "negative_reference_local_alignment": negative_step[
                    "local_target_alignment"
                ],
                "reference_shift_direction_cosine": shift_direction_cosine,
                "reference_contribution_identity_error": contribution_identity_error,
            }
        )
    return features, scores


def class_group_counts(y: np.ndarray, groups: np.ndarray, labels: Sequence[str]) -> dict[str, int]:
    return {
        label: len(set(groups[y == label].tolist()))
        for label in labels
    }


def grouped_splits(
    y: np.ndarray,
    groups: np.ndarray,
    labels: Sequence[str],
    requested_folds: int,
    seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, int]]:
    counts = class_group_counts(y, groups, labels)
    if any(count == 0 for count in counts.values()):
        raise ValueError(f"At least one requested class is absent: {counts}")
    fold_count = min(requested_folds, len(set(groups.tolist())), min(counts.values()))
    if fold_count < 2:
        raise ValueError(f"Need at least two distinct groups for every class: {counts}")
    splitter = StratifiedGroupKFold(n_splits=fold_count, shuffle=True, random_state=seed)
    splits = list(splitter.split(np.zeros(len(y)), y, groups))
    return splits, counts


def make_classifier(
    *,
    c_value: float,
    seed: int,
    max_iterations: int,
    standardize: bool,
):
    classifier = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        max_iter=max_iterations,
        solver="lbfgs",
        random_state=seed,
    )
    if standardize:
        return make_pipeline(StandardScaler(), classifier)
    return classifier


def metric_values(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[str]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            recall_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)
        ),
    }


def grouped_bootstrap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    labels: Sequence[str],
    replicates: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    point = metric_values(y_true, y_pred, labels)
    if replicates <= 0:
        return {name: {"value": value, "ci_low": value, "ci_high": value} for name, value in point.items()}
    unique_groups = np.asarray(sorted(set(groups.tolist())), dtype=object)
    indices_by_group = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {name: [] for name in point}
    for _ in range(replicates):
        selected_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        selected_indices = np.concatenate([indices_by_group[group] for group in selected_groups])
        values = metric_values(y_true[selected_indices], y_pred[selected_indices], labels)
        for name, value in values.items():
            samples[name].append(value)
    return {
        name: {
            "value": value,
            "ci_low": float(np.quantile(samples[name], 0.025)),
            "ci_high": float(np.quantile(samples[name], 0.975)),
        }
        for name, value in point.items()
    }


def select_c_inner(
    x_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    labels: Sequence[str],
    c_grid: Sequence[float],
    inner_folds: int,
    seed: int,
    max_iterations: int,
    standardize: bool,
) -> float:
    try:
        splits, _ = grouped_splits(y_train, groups_train, labels, inner_folds, seed)
    except ValueError:
        return float(c_grid[len(c_grid) // 2])
    candidates: list[tuple[float, float]] = []
    for c_value in c_grid:
        scores = []
        for train_index, validation_index in splits:
            model = make_classifier(
                c_value=c_value,
                seed=seed,
                max_iterations=max_iterations,
                standardize=standardize,
            )
            model.fit(x_train[train_index], y_train[train_index])
            prediction = model.predict(x_train[validation_index])
            scores.append(
                f1_score(
                    y_train[validation_index],
                    prediction,
                    labels=list(labels),
                    average="macro",
                    zero_division=0,
                )
            )
        candidates.append((float(np.mean(scores)), c_value))
    # Prefer the smaller C when validation scores tie.
    return float(max(candidates, key=lambda item: (item[0], -item[1]))[1])


def finalize_probe(
    name: str,
    y: np.ndarray,
    predictions: np.ndarray,
    groups: np.ndarray,
    labels: Sequence[str],
    class_counts: dict[str, int],
    class_group_count: dict[str, int],
    selected_c: Sequence[float],
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    metrics = grouped_bootstrap_metrics(
        y,
        predictions,
        groups,
        labels,
        bootstrap_replicates,
        seed,
    )
    matrix = confusion_matrix(y, predictions, labels=list(labels))
    return {
        "name": name,
        "status": "ok",
        "labels": list(labels),
        "row_count": len(y),
        "group_count": len(set(groups.tolist())),
        "class_counts": class_counts,
        "class_group_counts": class_group_count,
        "selected_c_by_fold": list(selected_c),
        "chance_uniform_accuracy": 1.0 / len(labels),
        "chance_majority_accuracy": max(class_counts.values()) / len(y),
        "metrics": metrics,
        "confusion_matrix": matrix.tolist(),
    }


def run_numeric_probe(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    labels: Sequence[str],
    *,
    folds: int,
    inner_folds: int,
    c_grid: Sequence[float],
    tune_c: bool,
    standardize: bool,
    bootstrap_replicates: int,
    seed: int,
    max_iterations: int,
) -> tuple[dict[str, Any], np.ndarray | None]:
    class_counts = {label: int(np.sum(y == label)) for label in labels}
    try:
        splits, group_counts = grouped_splits(y, groups, labels, folds, seed)
    except ValueError as exc:
        return (
            {
                "name": name,
                "status": "not_evaluable",
                "reason": str(exc),
                "row_count": len(y),
                "class_counts": class_counts,
            },
            None,
        )

    predictions = np.empty(len(y), dtype=object)
    selected_values: list[float] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        for fold_index, (train_index, test_index) in enumerate(splits):
            if set(y[train_index].tolist()) != set(labels):
                return (
                    {
                        "name": name,
                        "status": "not_evaluable",
                        "reason": f"outer fold {fold_index} training data is missing a class",
                        "class_counts": class_counts,
                    },
                    None,
                )
            c_value = (
                select_c_inner(
                    x[train_index],
                    y[train_index],
                    groups[train_index],
                    labels,
                    c_grid,
                    inner_folds,
                    seed + fold_index,
                    max_iterations,
                    standardize,
                )
                if tune_c
                else 1.0
            )
            selected_values.append(c_value)
            model = make_classifier(
                c_value=c_value,
                seed=seed + fold_index,
                max_iterations=max_iterations,
                standardize=standardize,
            )
            model.fit(x[train_index], y[train_index])
            predictions[test_index] = model.predict(x[test_index])

    report = finalize_probe(
        name,
        y,
        predictions,
        groups,
        labels,
        class_counts,
        group_counts,
        selected_values,
        bootstrap_replicates,
        seed,
    )
    report["outer_folds"] = len(splits)
    report["c_tuned_inside_outer_fold"] = tune_c
    return report, predictions


def run_text_probe(
    texts: Sequence[str],
    y: np.ndarray,
    groups: np.ndarray,
    labels: Sequence[str],
    *,
    folds: int,
    bootstrap_replicates: int,
    seed: int,
    max_iterations: int,
) -> tuple[dict[str, Any], np.ndarray | None]:
    name = "text_tfidf__cell"
    class_counts = {label: int(np.sum(y == label)) for label in labels}
    try:
        splits, group_counts = grouped_splits(y, groups, labels, folds, seed)
    except ValueError as exc:
        return {"name": name, "status": "not_evaluable", "reason": str(exc)}, None

    predictions = np.empty(len(y), dtype=object)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        for fold_index, (train_index, test_index) in enumerate(splits):
            model = make_pipeline(
                TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True),
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=max_iterations,
                    solver="lbfgs",
                    random_state=seed + fold_index,
                ),
            )
            model.fit([texts[index] for index in train_index], y[train_index])
            predictions[test_index] = model.predict([texts[index] for index in test_index])
    report = finalize_probe(
        name,
        y,
        predictions,
        groups,
        labels,
        class_counts,
        group_counts,
        [1.0] * len(splits),
        bootstrap_replicates,
        seed,
    )
    report["outer_folds"] = len(splits)
    report["input"] = "positive_text [SEP] negative_text"
    return report, predictions


def run_cross_mode_probe(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    modes: np.ndarray,
    labels: Sequence[str],
    train_mode: str,
    test_mode: str,
    *,
    folds: int,
    bootstrap_replicates: int,
    seed: int,
    max_iterations: int,
) -> tuple[dict[str, Any], np.ndarray | None, np.ndarray | None]:
    test_domain_indices = np.flatnonzero(modes == test_mode)
    y_test_domain = y[test_domain_indices]
    groups_test_domain = groups[test_domain_indices]
    try:
        domain_splits, group_counts = grouped_splits(
            y_test_domain, groups_test_domain, labels, folds, seed
        )
    except ValueError as exc:
        return {"name": name, "status": "not_evaluable", "reason": str(exc)}, None, None

    predictions = np.empty(len(test_domain_indices), dtype=object)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        for fold_index, (_, domain_test_local) in enumerate(domain_splits):
            fold_test_indices = test_domain_indices[domain_test_local]
            held_out_groups = set(groups[fold_test_indices].tolist())
            train_indices = np.asarray(
                [
                    index
                    for index in np.flatnonzero(modes == train_mode)
                    if groups[index] not in held_out_groups
                ],
                dtype=np.int64,
            )
            if set(y[train_indices].tolist()) != set(labels):
                return (
                    {
                        "name": name,
                        "status": "not_evaluable",
                        "reason": f"fold {fold_index} source mode is missing a class",
                    },
                    None,
                    None,
                )
            model = make_classifier(
                c_value=1.0,
                seed=seed + fold_index,
                max_iterations=max_iterations,
                standardize=False,
            )
            model.fit(x[train_indices], y[train_indices])
            predictions[domain_test_local] = model.predict(x[fold_test_indices])
    report = finalize_probe(
        name,
        y_test_domain,
        predictions,
        groups_test_domain,
        labels,
        {label: int(np.sum(y_test_domain == label)) for label in labels},
        group_counts,
        [1.0] * len(domain_splits),
        bootstrap_replicates,
        seed,
    )
    report.update(
        {
            "outer_folds": len(domain_splits),
            "train_mode": train_mode,
            "test_mode": test_mode,
            "source_and_test_groups_disjoint_per_fold": True,
        }
    )
    return report, predictions, test_domain_indices


def bootstrap_scalar_by_group(
    values: np.ndarray,
    groups: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    if replicates <= 0:
        point = statistic(values)
        return point, point
    unique_groups = np.asarray(sorted(set(groups.tolist())), dtype=object)
    indices_by_group = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(replicates):
        selected = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([indices_by_group[group] for group in selected])
        samples.append(statistic(values[indices]))
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def grouped_mean_with_bootstrap(
    values: np.ndarray,
    groups: np.ndarray,
    replicates: int,
    seed: int,
) -> tuple[float, float, float]:
    """Average within group first so every pair has equal cell-level weight."""

    unique_groups = np.asarray(sorted(set(groups.tolist())), dtype=object)
    group_means = np.asarray(
        [float(np.mean(values[groups == group])) for group in unique_groups],
        dtype=np.float64,
    )
    point = float(np.mean(group_means))
    if replicates <= 0:
        return point, point, point
    rng = np.random.default_rng(seed)
    samples = [
        float(np.mean(rng.choice(group_means, size=len(group_means), replace=True)))
        for _ in range(replicates)
    ]
    return point, float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def delta_cell_rows(
    scores: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    bootstrap_replicates: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    modes: Sequence[str] = (*CONSTRUCTION_MODES, "all")
    for mode_index, mode in enumerate(modes):
        for semantic_index, semantic in enumerate(SEMANTIC_TARGETS):
            for operation_index, operation in enumerate(EDIT_OPERATIONS):
                mask = (labels["semantic_target"] == semantic) & (
                    labels["edit_operation"] == operation
                )
                if mode != "all":
                    mask &= labels["construction_mode"] == mode
                indices = np.flatnonzero(mask)
                if not len(indices):
                    rows.append(
                        {
                            "construction_mode": mode,
                            "semantic_target": semantic,
                            "edit_operation": operation,
                            "count": 0,
                            "unique_pairs": 0,
                        }
                    )
                    continue
                delta = scores["delta"][indices]
                groups = labels["group"][indices]
                cell_seed = seed + mode_index * 100 + semantic_index * 10 + operation_index
                mean_ci = bootstrap_scalar_by_group(
                    delta,
                    groups,
                    lambda value: float(np.mean(value)),
                    bootstrap_replicates,
                    cell_seed,
                )
                win_ci = bootstrap_scalar_by_group(
                    delta,
                    groups,
                    lambda value: float(np.mean(value > 0)),
                    bootstrap_replicates,
                    cell_seed + 1000,
                )
                rows.append(
                    {
                        "construction_mode": mode,
                        "semantic_target": semantic,
                        "edit_operation": operation,
                        "count": len(indices),
                        "unique_pairs": len(set(groups.tolist())),
                        "mean_delta": float(np.mean(delta)),
                        "median_delta": float(np.median(delta)),
                        "std_delta": float(np.std(delta)),
                        "win_rate_delta_gt_0": float(np.mean(delta > 0)),
                        "mean_positive_score": float(
                            np.mean(scores["positive_score"][indices])
                        ),
                        "mean_negative_score": float(
                            np.mean(scores["negative_score"][indices])
                        ),
                        "mean_vector_norm": float(np.mean(scores["vector_norm"][indices])),
                        "mean_vector_target_cosine": float(
                            np.mean(scores["vector_target_cosine"][indices])
                        ),
                        "mean_query_angle_degrees": float(
                            np.mean(scores["query_angle_degrees"][indices])
                        ),
                        "mean_local_target_alignment": float(
                            np.mean(scores["local_target_alignment"][indices])
                        ),
                        "mean_delta_ci_low": mean_ci[0],
                        "mean_delta_ci_high": mean_ci[1],
                        "win_rate_ci_low": win_ci[0],
                        "win_rate_ci_high": win_ci[1],
                    }
                )
    return rows


def reference_ablation_cell_rows(
    scores: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    bootstrap_replicates: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Aggregate f(R), f(R,m+), and f(R,m-) contributions by 3x3 cell."""

    if "reference_score" not in scores:
        return []
    rows = []
    modes: Sequence[str] = (*CONSTRUCTION_MODES, "all")
    for mode_index, mode in enumerate(modes):
        for semantic_index, semantic in enumerate(SEMANTIC_TARGETS):
            for operation_index, operation in enumerate(EDIT_OPERATIONS):
                mask = (labels["semantic_target"] == semantic) & (
                    labels["edit_operation"] == operation
                )
                if mode != "all":
                    mask &= labels["construction_mode"] == mode
                indices = np.flatnonzero(mask)
                base_row: dict[str, Any] = {
                    "construction_mode": mode,
                    "semantic_target": semantic,
                    "edit_operation": operation,
                    "count": len(indices),
                    "unique_pairs": len(set(labels["group"][indices].tolist())),
                }
                if not len(indices):
                    rows.append(base_row)
                    continue

                groups = labels["group"][indices]
                positive_gain = scores["positive_gain_from_reference"][indices]
                negative_gain = scores["negative_gain_from_reference"][indices]
                delta = scores["delta"][indices]
                cell_seed = seed + mode_index * 100 + semantic_index * 10 + operation_index
                positive_pair = grouped_mean_with_bootstrap(
                    positive_gain,
                    groups,
                    bootstrap_replicates,
                    cell_seed + 2000,
                )
                negative_pair = grouped_mean_with_bootstrap(
                    negative_gain,
                    groups,
                    bootstrap_replicates,
                    cell_seed + 3000,
                )
                delta_pair = grouped_mean_with_bootstrap(
                    delta,
                    groups,
                    bootstrap_replicates,
                    cell_seed + 4000,
                )
                reference_pair = grouped_mean_with_bootstrap(
                    scores["reference_score"][indices], groups, 0, cell_seed
                )
                positive_score_pair = grouped_mean_with_bootstrap(
                    scores["positive_score"][indices], groups, 0, cell_seed
                )
                negative_score_pair = grouped_mean_with_bootstrap(
                    scores["negative_score"][indices], groups, 0, cell_seed
                )
                rows.append(
                    {
                        **base_row,
                        "mean_reference_score": float(
                            np.mean(scores["reference_score"][indices])
                        ),
                        "mean_positive_score": float(
                            np.mean(scores["positive_score"][indices])
                        ),
                        "mean_negative_score": float(
                            np.mean(scores["negative_score"][indices])
                        ),
                        "mean_positive_gain_from_reference": float(
                            np.mean(positive_gain)
                        ),
                        "median_positive_gain_from_reference": float(
                            np.median(positive_gain)
                        ),
                        "positive_gain_gt_0_rate": float(np.mean(positive_gain > 0)),
                        "mean_negative_gain_from_reference": float(
                            np.mean(negative_gain)
                        ),
                        "median_negative_gain_from_reference": float(
                            np.median(negative_gain)
                        ),
                        "negative_gain_gt_0_rate": float(np.mean(negative_gain > 0)),
                        "mean_delta": float(np.mean(delta)),
                        "pair_balanced_mean_reference_score": reference_pair[0],
                        "pair_balanced_mean_positive_score": positive_score_pair[0],
                        "pair_balanced_mean_negative_score": negative_score_pair[0],
                        "pair_balanced_mean_positive_gain_from_reference": positive_pair[0],
                        "pair_balanced_mean_negative_gain_from_reference": negative_pair[0],
                        "pair_balanced_mean_delta": delta_pair[0],
                        "mean_positive_reference_distance": float(
                            np.mean(scores["positive_reference_distance"][indices])
                        ),
                        "mean_negative_reference_distance": float(
                            np.mean(scores["negative_reference_distance"][indices])
                        ),
                        "mean_positive_reference_angle_degrees": float(
                            np.mean(scores["positive_reference_angle_degrees"][indices])
                        ),
                        "mean_negative_reference_angle_degrees": float(
                            np.mean(scores["negative_reference_angle_degrees"][indices])
                        ),
                        "mean_positive_reference_target_cosine": float(
                            np.mean(scores["positive_reference_target_cosine"][indices])
                        ),
                        "mean_negative_reference_target_cosine": float(
                            np.mean(scores["negative_reference_target_cosine"][indices])
                        ),
                        "mean_positive_reference_local_alignment": float(
                            np.mean(scores["positive_reference_local_alignment"][indices])
                        ),
                        "mean_negative_reference_local_alignment": float(
                            np.mean(scores["negative_reference_local_alignment"][indices])
                        ),
                        "mean_reference_shift_direction_cosine": float(
                            np.mean(scores["reference_shift_direction_cosine"][indices])
                        ),
                        "pair_balanced_positive_gain_ci_low": positive_pair[1],
                        "pair_balanced_positive_gain_ci_high": positive_pair[2],
                        "pair_balanced_negative_gain_ci_low": negative_pair[1],
                        "pair_balanced_negative_gain_ci_high": negative_pair[2],
                        "pair_balanced_delta_ci_low": delta_pair[1],
                        "pair_balanced_delta_ci_high": delta_pair[2],
                    }
                )
    return rows


def design_matrix(
    semantic: np.ndarray,
    operation: np.ndarray,
    *,
    saturated: bool,
) -> np.ndarray:
    columns = [np.ones(len(semantic), dtype=np.float64)]
    if saturated:
        cells = np.asarray(
            [f"{target}|{edit}" for target, edit in zip(semantic, operation)],
            dtype=object,
        )
        columns.extend((cells == cell).astype(np.float64) for cell in CELL_LABELS[:-1])
    else:
        columns.extend((semantic == value).astype(np.float64) for value in SEMANTIC_TARGETS[:-1])
        columns.extend((operation == value).astype(np.float64) for value in EDIT_OPERATIONS[:-1])
    return np.column_stack(columns)


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    penalty = np.eye(x.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def factorial_structure(
    vector: np.ndarray,
    labels: dict[str, np.ndarray],
    folds: int,
    seed: int,
    alpha: float,
) -> dict[str, Any]:
    cell = labels["cell"]
    groups = labels["group"]
    try:
        splits, group_counts = grouped_splits(cell, groups, CELL_LABELS, folds, seed)
    except ValueError as exc:
        return {"status": "not_evaluable", "reason": str(exc)}

    results: dict[str, Any] = {}
    predictions_by_model: dict[str, np.ndarray] = {}
    for model_name, saturated in (("additive", False), ("saturated_9_cell", True)):
        predictions = np.zeros_like(vector, dtype=np.float64)
        baseline_denominator = 0.0
        for train_index, test_index in splits:
            x_train = design_matrix(
                labels["semantic_target"][train_index],
                labels["edit_operation"][train_index],
                saturated=saturated,
            )
            x_test = design_matrix(
                labels["semantic_target"][test_index],
                labels["edit_operation"][test_index],
                saturated=saturated,
            )
            coefficients = ridge_fit(x_train, vector[train_index].astype(np.float64), alpha)
            predictions[test_index] = x_test @ coefficients
            train_mean = vector[train_index].mean(axis=0, dtype=np.float64)
            baseline_denominator += float(
                np.square(vector[test_index].astype(np.float64) - train_mean).sum()
            )
        residual_sum = float(np.square(vector.astype(np.float64) - predictions).sum())
        prediction_norm = np.linalg.norm(predictions, axis=1)
        vector_norm = np.linalg.norm(vector, axis=1)
        cosine = np.divide(
            np.einsum("nd,nd->n", predictions, vector),
            prediction_norm * vector_norm,
            out=np.zeros(len(vector), dtype=np.float64),
            where=(prediction_norm * vector_norm) > 1e-12,
        )
        results[model_name] = {
            "frobenius_r2_against_fold_train_mean": 1.0 - residual_sum / baseline_denominator,
            "mean_prediction_cosine": float(np.mean(cosine)),
            "residual_sum_squares": residual_sum,
        }
        predictions_by_model[model_name] = predictions
    results["interaction_gain_r2"] = (
        results["saturated_9_cell"]["frobenius_r2_against_fold_train_mean"]
        - results["additive"]["frobenius_r2_against_fold_train_mean"]
    )
    return {
        "status": "ok",
        "outer_folds": len(splits),
        "grouped_by_pair_id": True,
        "class_group_counts": group_counts,
        "ridge_alpha": alpha,
        **results,
    }


def write_csv_atomic(path: Path, rows: Iterable[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_confusion_csv(path: Path, report: dict[str, Any]) -> None:
    if report.get("status") != "ok":
        return
    labels = report["labels"]
    rows = []
    for actual, counts in zip(labels, report["confusion_matrix"]):
        rows.append({"actual": actual, **{predicted: count for predicted, count in zip(labels, counts)}})
    write_csv_atomic(path, rows, ["actual", *labels])


def flatten_probe_metrics(probes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for probe in probes:
        if probe.get("status") != "ok":
            rows.append(
                {
                    "probe": probe["name"],
                    "status": probe["status"],
                    "reason": probe.get("reason", ""),
                }
            )
            continue
        row: dict[str, Any] = {
            "probe": probe["name"],
            "status": "ok",
            "rows": probe["row_count"],
            "groups": probe["group_count"],
            "outer_folds": probe.get("outer_folds"),
            "uniform_chance": probe["chance_uniform_accuracy"],
            "majority_chance": probe["chance_majority_accuracy"],
        }
        for metric_name, values in probe["metrics"].items():
            row[metric_name] = values["value"]
            row[f"{metric_name}_ci_low"] = values["ci_low"]
            row[f"{metric_name}_ci_high"] = values["ci_high"]
        rows.append(row)
    return rows


def make_summary(
    extraction_metadata: dict[str, Any],
    probes: Sequence[dict[str, Any]],
    factor_report: dict[str, Any],
    reference_ablation_available: bool,
) -> str:
    lines = [
        f"# {extraction_metadata.get('model_name', 'model')} 3×3 diagnostic",
        "",
        f"- Samples: {extraction_metadata.get('sample_count')}",
        f"- Negatives: {extraction_metadata.get('negative_count')}",
        f"- Embedding dimension: {extraction_metadata.get('embedding_dimension')}",
        "- All probe splits are grouped by pair_id.",
        (
            "- Reference ablation f_query(R, empty) is available in "
            "reference_ablation_by_cell.csv."
            if reference_ablation_available
            else "- Reference ablation is unavailable; rerun extraction to add q_reference."
        ),
        "",
        "## Linear probes",
        "",
        "| Probe | Accuracy | Macro-F1 | Balanced accuracy |",
        "|---|---:|---:|---:|",
    ]
    for probe in probes:
        if probe.get("status") != "ok":
            lines.append(f"| {probe['name']} | not evaluable | — | — |")
            continue
        metrics = probe["metrics"]
        lines.append(
            f"| {probe['name']} | {metrics['accuracy']['value']:.4f} | "
            f"{metrics['macro_f1']['value']:.4f} | "
            f"{metrics['balanced_accuracy']['value']:.4f} |"
        )
    lines.extend(["", "## Factorial structure", ""])
    if factor_report.get("status") == "ok":
        lines.extend(
            [
                f"- Additive held-out R²: {factor_report['additive']['frobenius_r2_against_fold_train_mean']:.4f}",
                f"- Saturated 9-cell held-out R²: {factor_report['saturated_9_cell']['frobenius_r2_against_fold_train_mean']:.4f}",
                f"- Interaction gain: {factor_report['interaction_gain_r2']:.4f}",
            ]
        )
    else:
        lines.append(f"Not evaluable: {factor_report.get('reason', 'unknown reason')}")
    lines.extend(
        [
            "",
            "Delete negatives are weaker queries and are not required to contradict the target; ",
            "interpret their delta descriptively rather than as strict retrieval errors.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.folds < 2 or args.inner_folds < 2:
        raise ValueError("--folds and --inner-folds must both be at least 2")
    c_grid = tuple(float(value) for value in args.c_grid.split(",") if value.strip())
    if not c_grid or any(value <= 0 for value in c_grid):
        raise ValueError("--c-grid must contain positive comma-separated values")

    arrays, rows, extraction_metadata = load_features(args.features_dir)
    features, scores = make_feature_views(arrays)
    labels = label_arrays(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[info] model={extraction_metadata.get('model_name')} rows={len(rows)} "
        f"groups={len(set(labels['group'].tolist()))} dimension={features['raw_v'].shape[1]}"
    )

    delta_rows = delta_cell_rows(
        scores,
        labels,
        args.bootstrap_replicates,
        args.seed,
    )
    delta_fields = [
        "construction_mode",
        "semantic_target",
        "edit_operation",
        "count",
        "unique_pairs",
        "mean_delta",
        "median_delta",
        "std_delta",
        "win_rate_delta_gt_0",
        "mean_positive_score",
        "mean_negative_score",
        "mean_vector_norm",
        "mean_vector_target_cosine",
        "mean_query_angle_degrees",
        "mean_local_target_alignment",
        "mean_delta_ci_low",
        "mean_delta_ci_high",
        "win_rate_ci_low",
        "win_rate_ci_high",
    ]
    write_csv_atomic(args.output_dir / "delta_by_cell.csv", delta_rows, delta_fields)

    reference_rows = reference_ablation_cell_rows(
        scores,
        labels,
        args.bootstrap_replicates,
        args.seed,
    )
    reference_fields = [
        "construction_mode",
        "semantic_target",
        "edit_operation",
        "count",
        "unique_pairs",
        "mean_reference_score",
        "mean_positive_score",
        "mean_negative_score",
        "mean_positive_gain_from_reference",
        "median_positive_gain_from_reference",
        "positive_gain_gt_0_rate",
        "mean_negative_gain_from_reference",
        "median_negative_gain_from_reference",
        "negative_gain_gt_0_rate",
        "mean_delta",
        "pair_balanced_mean_reference_score",
        "pair_balanced_mean_positive_score",
        "pair_balanced_mean_negative_score",
        "pair_balanced_mean_positive_gain_from_reference",
        "pair_balanced_mean_negative_gain_from_reference",
        "pair_balanced_mean_delta",
        "mean_positive_reference_distance",
        "mean_negative_reference_distance",
        "mean_positive_reference_angle_degrees",
        "mean_negative_reference_angle_degrees",
        "mean_positive_reference_target_cosine",
        "mean_negative_reference_target_cosine",
        "mean_positive_reference_local_alignment",
        "mean_negative_reference_local_alignment",
        "mean_reference_shift_direction_cosine",
        "pair_balanced_positive_gain_ci_low",
        "pair_balanced_positive_gain_ci_high",
        "pair_balanced_negative_gain_ci_low",
        "pair_balanced_negative_gain_ci_high",
        "pair_balanced_delta_ci_low",
        "pair_balanced_delta_ci_high",
    ]
    if reference_rows:
        write_csv_atomic(
            args.output_dir / "reference_ablation_by_cell.csv",
            reference_rows,
            reference_fields,
        )

    if args.descriptive_only:
        descriptive_report = {
            "schema_version": "1.0",
            "extraction": extraction_metadata,
            "analysis": {
                "descriptive_only": True,
                "bootstrap_replicates": args.bootstrap_replicates,
                "seed": args.seed,
                "grouped_by": "pair_id",
                "delta_identity_max_abs_error": float(
                    scores["identity_error"].max(initial=0.0)
                ),
                "reference_ablation_available": bool(reference_rows),
                "reference_contribution_identity_max_abs_error": (
                    float(
                        scores["reference_contribution_identity_error"].max(
                            initial=0.0
                        )
                    )
                    if reference_rows
                    else None
                ),
                "cell_counts": dict(Counter(labels["cell"].tolist())),
                "mode_counts": dict(Counter(labels["construction_mode"].tolist())),
            },
            "files": {
                "delta_by_cell": "delta_by_cell.csv",
                "reference_ablation_by_cell": (
                    "reference_ablation_by_cell.csv" if reference_rows else None
                ),
            },
        }
        write_json_atomic(
            args.output_dir / "descriptive_report.json", descriptive_report
        )
        print(
            "[done] descriptive statistics refreshed; "
            "linear probes and factor models were not rerun"
        )
        return

    probe_specs = [
        ("raw_v__cell", "raw_v", "cell", CELL_LABELS, True, False, None),
        (
            "raw_v__semantic_target",
            "raw_v",
            "semantic_target",
            SEMANTIC_TARGETS,
            True,
            False,
            None,
        ),
        (
            "raw_v__edit_operation",
            "raw_v",
            "edit_operation",
            EDIT_OPERATIONS,
            True,
            False,
            None,
        ),
        ("direction_v__cell", "direction_v", "cell", CELL_LABELS, True, False, None),
        ("norm_v__cell", "norm_v", "cell", CELL_LABELS, False, True, None),
        ("q_positive__cell", "q_positive", "cell", CELL_LABELS, False, False, None),
        ("q_negative__cell", "q_negative", "cell", CELL_LABELS, False, False, None),
        (
            "raw_v__construction_mode",
            "raw_v",
            "construction_mode",
            CONSTRUCTION_MODES,
            False,
            False,
            None,
        ),
        (
            "raw_v__cell__text_only",
            "raw_v",
            "cell",
            CELL_LABELS,
            False,
            False,
            "text_only_editable",
        ),
        (
            "raw_v__cell__vision_grounded",
            "raw_v",
            "cell",
            CELL_LABELS,
            False,
            False,
            "vision_grounded",
        ),
    ]
    probes: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for name, feature_name, label_name, class_labels, tune_c, standardize, mode in probe_specs:
        mask = np.ones(len(rows), dtype=bool)
        if mode is not None:
            mask &= labels["construction_mode"] == mode
        selected = np.flatnonzero(mask)
        report, predictions = run_numeric_probe(
            name,
            features[feature_name][selected],
            labels[label_name][selected],
            labels["group"][selected],
            class_labels,
            folds=args.folds,
            inner_folds=args.inner_folds,
            c_grid=c_grid,
            tune_c=tune_c,
            standardize=standardize,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
            max_iterations=args.max_iterations,
        )
        probes.append(report)
        write_confusion_csv(args.output_dir / f"confusion_{name}.csv", report)
        if predictions is not None:
            for local_index, prediction in enumerate(predictions):
                global_index = int(selected[local_index])
                prediction_rows.append(
                    {
                        "probe": name,
                        "negative_id": rows[global_index]["negative_id"],
                        "pair_id": rows[global_index]["pair_id"],
                        "true_label": str(labels[label_name][global_index]),
                        "prediction": str(prediction),
                    }
                )
        status = report["status"]
        accuracy = (
            f" accuracy={report['metrics']['accuracy']['value']:.4f}"
            if status == "ok"
            else f" reason={report.get('reason')}"
        )
        print(f"[probe] {name}: {status}{accuracy}")

    text_inputs = [
        f"{row['positive_text']} [SEP] {row['negative_text']}"
        for row in rows
    ]
    text_report, text_predictions = run_text_probe(
        text_inputs,
        labels["cell"],
        labels["group"],
        CELL_LABELS,
        folds=args.folds,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
        max_iterations=args.max_iterations,
    )
    probes.append(text_report)
    write_confusion_csv(args.output_dir / "confusion_text_tfidf__cell.csv", text_report)
    if text_predictions is not None:
        for index, prediction in enumerate(text_predictions):
            prediction_rows.append(
                {
                    "probe": text_report["name"],
                    "negative_id": rows[index]["negative_id"],
                    "pair_id": rows[index]["pair_id"],
                    "true_label": str(labels["cell"][index]),
                    "prediction": str(prediction),
                }
            )

    for label_name, class_labels in (
        ("semantic_target", SEMANTIC_TARGETS),
        ("edit_operation", EDIT_OPERATIONS),
    ):
        for train_mode, test_mode in (
            ("text_only_editable", "vision_grounded"),
            ("vision_grounded", "text_only_editable"),
        ):
            name = f"raw_v__{label_name}__{train_mode}_to_{test_mode}"
            report, predictions, global_indices = run_cross_mode_probe(
                name,
                features["raw_v"],
                labels[label_name],
                labels["group"],
                labels["construction_mode"],
                class_labels,
                train_mode,
                test_mode,
                folds=args.folds,
                bootstrap_replicates=args.bootstrap_replicates,
                seed=args.seed,
                max_iterations=args.max_iterations,
            )
            probes.append(report)
            write_confusion_csv(args.output_dir / f"confusion_{name}.csv", report)
            if predictions is not None and global_indices is not None:
                for local_index, prediction in enumerate(predictions):
                    global_index = int(global_indices[local_index])
                    prediction_rows.append(
                        {
                            "probe": name,
                            "negative_id": rows[global_index]["negative_id"],
                            "pair_id": rows[global_index]["pair_id"],
                            "true_label": str(labels[label_name][global_index]),
                            "prediction": str(prediction),
                        }
                    )

    factor_report = factorial_structure(
        features["raw_v"],
        labels,
        args.folds,
        args.seed,
        args.factor_ridge,
    )
    write_json_atomic(args.output_dir / "factorial_report.json", factor_report)
    write_jsonl_atomic(args.output_dir / "oof_predictions.jsonl", prediction_rows)

    metric_rows = flatten_probe_metrics(probes)
    metric_fields = [
        "probe",
        "status",
        "reason",
        "rows",
        "groups",
        "outer_folds",
        "uniform_chance",
        "majority_chance",
        "accuracy",
        "accuracy_ci_low",
        "accuracy_ci_high",
        "macro_f1",
        "macro_f1_ci_low",
        "macro_f1_ci_high",
        "balanced_accuracy",
        "balanced_accuracy_ci_low",
        "balanced_accuracy_ci_high",
    ]
    write_csv_atomic(args.output_dir / "probe_metrics.csv", metric_rows, metric_fields)

    report = {
        "schema_version": "1.0",
        "extraction": extraction_metadata,
        "analysis": {
            "folds_requested": args.folds,
            "inner_folds_requested": args.inner_folds,
            "c_grid": c_grid,
            "bootstrap_replicates": args.bootstrap_replicates,
            "seed": args.seed,
            "grouped_by": "pair_id",
            "delta_identity_max_abs_error": float(scores["identity_error"].max(initial=0.0)),
            "reference_ablation_available": bool(reference_rows),
            "reference_contribution_identity_max_abs_error": (
                float(
                    scores["reference_contribution_identity_error"].max(initial=0.0)
                )
                if reference_rows
                else None
            ),
            "cell_counts": dict(Counter(labels["cell"].tolist())),
            "mode_counts": dict(Counter(labels["construction_mode"].tolist())),
        },
        "probes": probes,
        "factorial_structure": factor_report,
    }
    write_json_atomic(args.output_dir / "report.json", report)
    (args.output_dir / "summary.md").write_text(
        make_summary(
            extraction_metadata,
            probes,
            factor_report,
            reference_ablation_available=bool(reference_rows),
        ),
        encoding="utf-8",
    )
    print(f"[done] report={args.output_dir / 'report.json'}")


if __name__ == "__main__":
    main()
