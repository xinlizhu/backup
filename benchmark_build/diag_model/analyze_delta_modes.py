#!/usr/bin/env python3
"""Compare Delta distributions for text-only and vision-grounded negatives.

The script is deliberately offline: it reads cached q+/q-/t arrays, writes
pair-aware comparison tables, and renders count histograms and win-rate plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cir_qdiag_matplotlib")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import (  # noqa: E402
    CELL_LABELS,
    CONSTRUCTION_MODES,
    EDIT_OPERATIONS,
    SEMANTIC_TARGETS,
    read_jsonl,
    write_json_atomic,
)


MODEL_ORDER = (
    "BGE-VL-base",
    "BGE-VL-MLLM-S1",
    "Qwen3-VL-Embedding-2B",
)
MODEL_SHORT = {
    "BGE-VL-base": "BGE-base",
    "BGE-VL-MLLM-S1": "MLLM-S1",
    "Qwen3-VL-Embedding-2B": "Qwen3-VL-2B",
}
MODE_COLORS = {
    "text_only_editable": "#4C78A8",
    "vision_grounded": "#F58518",
}
MODE_LABELS = {
    "text_only_editable": "text-only",
    "vision_grounded": "vision-grounded",
}


@dataclass(frozen=True)
class ModelData:
    name: str
    delta: np.ndarray
    mode: np.ndarray
    semantic: np.ndarray
    operation: np.ndarray
    pair: np.ndarray
    cell: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze and plot Delta by construction mode and 3x3 cell."
    )
    parser.add_argument(
        "--diagnostics-root",
        required=True,
        type=Path,
        help="Directory containing one diagnostics subdirectory per model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <diagnostics-root>/delta_mode_analysis.",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def load_model_data(model_dir: Path) -> ModelData:
    feature_path = model_dir / "features.npz"
    rows_path = model_dir / "rows.jsonl"
    if not feature_path.is_file() or not rows_path.is_file():
        raise FileNotFoundError(f"Missing diagnostic features under {model_dir}")
    with np.load(feature_path, allow_pickle=False) as archive:
        q_positive = np.asarray(archive["q_positive"], dtype=np.float32)
        q_negative = np.asarray(archive["q_negative"], dtype=np.float32)
        target = np.asarray(archive["target"], dtype=np.float32)
        sample_index = np.asarray(archive["sample_index"], dtype=np.int64)
    rows = list(read_jsonl(rows_path))
    rows.sort(key=lambda row: int(row["negative_index"]))
    if len(rows) != len(q_negative) or sample_index.shape != (len(rows),):
        raise ValueError(f"Inconsistent rows/features under {model_dir}")
    positive = q_positive[sample_index]
    target_rows = target[sample_index]
    positive_score = np.einsum("nd,nd->n", positive, target_rows)
    negative_score = np.einsum("nd,nd->n", q_negative, target_rows)
    delta = positive_score - negative_score
    if float(delta.min(initial=0.0)) < -2.00001 or float(delta.max(initial=0.0)) > 2.00001:
        raise AssertionError("Cosine difference must lie in [-2, 2]")
    semantic = np.asarray([row["semantic_target"] for row in rows], dtype=object)
    operation = np.asarray([row["edit_operation"] for row in rows], dtype=object)
    return ModelData(
        name=model_dir.name,
        delta=delta,
        mode=np.asarray([row["construction_mode"] for row in rows], dtype=object),
        semantic=semantic,
        operation=operation,
        pair=np.asarray([str(row["pair_id"]) for row in rows], dtype=object),
        cell=np.asarray(
            [f"{semantic_value}|{operation_value}" for semantic_value, operation_value in zip(semantic, operation)],
            dtype=object,
        ),
    )


def write_csv_atomic(
    path: Path,
    rows: Iterable[dict[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def quantile_interval(values: np.ndarray) -> tuple[float, float]:
    return float(np.nanquantile(values, 0.025)), float(np.nanquantile(values, 0.975))


def bootstrap_mean(
    values: np.ndarray,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    point = float(np.mean(values))
    if replicates <= 0:
        return point, point
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(values), size=(replicates, len(values)))
    samples = np.mean(values[selected], axis=1)
    return quantile_interval(samples)


def wilson_interval(successes: int, count: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if count <= 0:
        return float("nan"), float("nan")
    probability = successes / count
    denominator = 1.0 + z * z / count
    center = (probability + z * z / (2.0 * count)) / denominator
    half_width = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / count
            + z * z / (4.0 * count * count)
        )
        / denominator
    )
    return center - half_width, center + half_width


def pair_mode_values(
    data: ModelData,
    mask: np.ndarray,
    mode: str,
    transform: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    selected = mask & (data.mode == mode)
    groups = np.asarray(sorted(set(data.pair[selected].tolist())), dtype=object)
    values = np.asarray(
        [float(np.mean(transform(data.delta[selected & (data.pair == group)]))) for group in groups],
        dtype=np.float64,
    )
    return groups, values


def joint_pair_bootstrap(
    data: ModelData,
    mask: np.ndarray,
    transform: Callable[[np.ndarray], np.ndarray],
    replicates: int,
    seed: int,
) -> dict[str, float]:
    groups = np.asarray(sorted(set(data.pair[mask].tolist())), dtype=object)
    by_mode = {}
    for mode in CONSTRUCTION_MODES:
        mode_groups, mode_values = pair_mode_values(data, mask, mode, transform)
        mapping = dict(zip(mode_groups.tolist(), mode_values.tolist()))
        by_mode[mode] = np.asarray([mapping.get(group, np.nan) for group in groups])

    text = by_mode["text_only_editable"]
    vision = by_mode["vision_grounded"]
    point_text = float(np.nanmean(text))
    point_vision = float(np.nanmean(vision))
    point_difference = point_vision - point_text
    if replicates <= 0:
        return {
            "text": point_text,
            "vision": point_vision,
            "difference": point_difference,
            "ci_low": point_difference,
            "ci_high": point_difference,
        }
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(groups), size=(replicates, len(groups)))
    text_samples = text[selected]
    vision_samples = vision[selected]
    text_counts = np.sum(np.isfinite(text_samples), axis=1)
    vision_counts = np.sum(np.isfinite(vision_samples), axis=1)
    text_means = np.divide(
        np.nansum(text_samples, axis=1),
        text_counts,
        out=np.full(replicates, np.nan, dtype=np.float64),
        where=text_counts > 0,
    )
    vision_means = np.divide(
        np.nansum(vision_samples, axis=1),
        vision_counts,
        out=np.full(replicates, np.nan, dtype=np.float64),
        where=vision_counts > 0,
    )
    differences = vision_means - text_means
    ci_low, ci_high = quantile_interval(differences)
    return {
        "text": point_text,
        "vision": point_vision,
        "difference": point_difference,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def pair_balanced_comparison(
    data: ModelData,
    mask: np.ndarray,
    scope: str,
    semantic: str,
    operation: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    win = joint_pair_bootstrap(
        data,
        mask,
        lambda values: (values > 0).astype(np.float64),
        replicates,
        seed,
    )
    mean_delta = joint_pair_bootstrap(
        data,
        mask,
        lambda values: values.astype(np.float64),
        replicates,
        seed + 10000,
    )
    text_mask = mask & (data.mode == "text_only_editable")
    vision_mask = mask & (data.mode == "vision_grounded")
    return {
        "model": data.name,
        "comparison": "pair_balanced",
        "scope": scope,
        "semantic_target": semantic,
        "edit_operation": operation,
        "text_rows": int(np.sum(text_mask)),
        "vision_rows": int(np.sum(vision_mask)),
        "text_pairs": len(set(data.pair[text_mask].tolist())),
        "vision_pairs": len(set(data.pair[vision_mask].tolist())),
        "matched_pair_cells": "",
        "matched_pairs": "",
        "text_win_rate": win["text"],
        "vision_win_rate": win["vision"],
        "win_rate_diff_vision_minus_text": win["difference"],
        "win_rate_diff_ci_low": win["ci_low"],
        "win_rate_diff_ci_high": win["ci_high"],
        "text_mean_delta": mean_delta["text"],
        "vision_mean_delta": mean_delta["vision"],
        "mean_delta_diff_vision_minus_text": mean_delta["difference"],
        "mean_delta_diff_ci_low": mean_delta["ci_low"],
        "mean_delta_diff_ci_high": mean_delta["ci_high"],
    }


def cell_macro_comparison(
    data: ModelData,
    mask: np.ndarray,
    cells: Sequence[str],
    scope: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    groups = np.asarray(sorted(set(data.pair[mask].tolist())), dtype=object)

    def matrix(mode: str, use_win: bool) -> np.ndarray:
        result = np.full((len(groups), len(cells)), np.nan, dtype=np.float64)
        for group_index, group in enumerate(groups):
            for cell_index, cell in enumerate(cells):
                selected = (
                    mask
                    & (data.pair == group)
                    & (data.cell == cell)
                    & (data.mode == mode)
                )
                if np.any(selected):
                    values = data.delta[selected]
                    result[group_index, cell_index] = float(
                        np.mean(values > 0) if use_win else np.mean(values)
                    )
        return result

    matrices = {
        (mode, use_win): matrix(mode, use_win)
        for mode in CONSTRUCTION_MODES
        for use_win in (False, True)
    }

    def macro_value(value: np.ndarray) -> float:
        return float(np.mean([np.nanmean(value[:, index]) for index in range(value.shape[1])]))

    points = {}
    for use_win, prefix in ((True, "win"), (False, "delta")):
        text = matrices[("text_only_editable", use_win)]
        vision = matrices[("vision_grounded", use_win)]
        text_point = macro_value(text)
        vision_point = macro_value(vision)
        points[prefix] = (text_point, vision_point, vision_point - text_point)

    distributions: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(seed)
    for use_win, prefix in ((True, "win"), (False, "delta")):
        text = matrices[("text_only_editable", use_win)]
        vision = matrices[("vision_grounded", use_win)]
        differences = np.empty(replicates, dtype=np.float64)
        for start in range(0, replicates, 256):
            end = min(start + 256, replicates)
            selected = rng.integers(
                0, len(groups), size=(end - start, len(groups))
            )

            def bootstrap_macro(value: np.ndarray) -> np.ndarray:
                sample = value[selected]
                counts = np.sum(np.isfinite(sample), axis=1)
                cell_means = np.divide(
                    np.nansum(sample, axis=1),
                    counts,
                    out=np.full(counts.shape, np.nan, dtype=np.float64),
                    where=counts > 0,
                )
                complete = np.all(np.isfinite(cell_means), axis=1)
                result = np.full(end - start, np.nan, dtype=np.float64)
                result[complete] = np.mean(cell_means[complete], axis=1)
                return result

            differences[start:end] = bootstrap_macro(vision) - bootstrap_macro(text)
        distributions[prefix] = differences
    win_ci = quantile_interval(distributions["win"])
    delta_ci = quantile_interval(distributions["delta"])
    text_mask = mask & (data.mode == "text_only_editable")
    vision_mask = mask & (data.mode == "vision_grounded")
    return {
        "model": data.name,
        "comparison": "cell_macro_pair_bootstrap",
        "scope": scope,
        "semantic_target": "all_cells_equal_weight",
        "edit_operation": "strict" if scope == "strict" else "all",
        "text_rows": int(np.sum(text_mask)),
        "vision_rows": int(np.sum(vision_mask)),
        "text_pairs": len(set(data.pair[text_mask].tolist())),
        "vision_pairs": len(set(data.pair[vision_mask].tolist())),
        "matched_pair_cells": "",
        "matched_pairs": "",
        "text_win_rate": points["win"][0],
        "vision_win_rate": points["win"][1],
        "win_rate_diff_vision_minus_text": points["win"][2],
        "win_rate_diff_ci_low": win_ci[0],
        "win_rate_diff_ci_high": win_ci[1],
        "text_mean_delta": points["delta"][0],
        "vision_mean_delta": points["delta"][1],
        "mean_delta_diff_vision_minus_text": points["delta"][2],
        "mean_delta_diff_ci_low": delta_ci[0],
        "mean_delta_diff_ci_high": delta_ci[1],
    }


def matched_pair_cell_comparison(
    data: ModelData,
    mask: np.ndarray,
    scope: str,
    semantic: str,
    operation: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    selected_indices = np.flatnonzero(mask)
    unit_values: dict[tuple[str, str], dict[str, float]] = {}
    for index in selected_indices:
        key = (str(data.pair[index]), str(data.cell[index]))
        unit_values.setdefault(key, {})[str(data.mode[index])] = float(data.delta[index])
    matched = {
        key: value
        for key, value in unit_values.items()
        if all(mode in value for mode in CONSTRUCTION_MODES)
    }
    matched_pairs = sorted(set(pair for pair, _ in matched))
    if not matched_pairs:
        raise ValueError(f"No matched pair-cell units for {data.name} {scope}")

    pair_values = []
    for pair in matched_pairs:
        values = [value for (unit_pair, _), value in matched.items() if unit_pair == pair]
        text_delta = float(np.mean([value["text_only_editable"] for value in values]))
        vision_delta = float(np.mean([value["vision_grounded"] for value in values]))
        text_win = float(
            np.mean([value["text_only_editable"] > 0 for value in values])
        )
        vision_win = float(
            np.mean([value["vision_grounded"] > 0 for value in values])
        )
        pair_values.append((text_win, vision_win, text_delta, vision_delta))
    array = np.asarray(pair_values, dtype=np.float64)
    point = np.mean(array, axis=0)
    rng = np.random.default_rng(seed)
    chosen = rng.integers(0, len(array), size=(replicates, len(array)))
    samples = np.mean(array[chosen], axis=1)
    win_differences = samples[:, 1] - samples[:, 0]
    delta_differences = samples[:, 3] - samples[:, 2]
    win_ci = quantile_interval(win_differences)
    delta_ci = quantile_interval(delta_differences)
    text_mask = mask & (data.mode == "text_only_editable")
    vision_mask = mask & (data.mode == "vision_grounded")
    return {
        "model": data.name,
        "comparison": "matched_pair_cell",
        "scope": scope,
        "semantic_target": semantic,
        "edit_operation": operation,
        "text_rows": int(np.sum(text_mask)),
        "vision_rows": int(np.sum(vision_mask)),
        "text_pairs": len(set(data.pair[text_mask].tolist())),
        "vision_pairs": len(set(data.pair[vision_mask].tolist())),
        "matched_pair_cells": len(matched),
        "matched_pairs": len(matched_pairs),
        "text_win_rate": float(point[0]),
        "vision_win_rate": float(point[1]),
        "win_rate_diff_vision_minus_text": float(point[1] - point[0]),
        "win_rate_diff_ci_low": win_ci[0],
        "win_rate_diff_ci_high": win_ci[1],
        "text_mean_delta": float(point[2]),
        "vision_mean_delta": float(point[3]),
        "mean_delta_diff_vision_minus_text": float(point[3] - point[2]),
        "mean_delta_diff_ci_low": delta_ci[0],
        "mean_delta_diff_ci_high": delta_ci[1],
    }


def mode_win_rate_rows(
    data: ModelData,
    replicates: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    for semantic_index, semantic in enumerate(SEMANTIC_TARGETS):
        for operation_index, operation in enumerate(EDIT_OPERATIONS):
            for mode_index, mode in enumerate(CONSTRUCTION_MODES):
                mask = (
                    (data.semantic == semantic)
                    & (data.operation == operation)
                    & (data.mode == mode)
                )
                groups, win_values = pair_mode_values(
                    data,
                    mask,
                    mode,
                    lambda values: (values > 0).astype(np.float64),
                )
                _, delta_values = pair_mode_values(
                    data,
                    mask,
                    mode,
                    lambda values: values.astype(np.float64),
                )
                win_ci = bootstrap_mean(
                    win_values,
                    replicates,
                    seed + semantic_index * 100 + operation_index * 10 + mode_index,
                )
                delta_ci = bootstrap_mean(
                    delta_values,
                    replicates,
                    seed + 1000 + semantic_index * 100 + operation_index * 10 + mode_index,
                )
                successes = int(np.sum(data.delta[mask] > 0))
                wilson_ci = wilson_interval(successes, int(np.sum(mask)))
                rows.append(
                    {
                        "model": data.name,
                        "construction_mode": mode,
                        "semantic_target": semantic,
                        "edit_operation": operation,
                        "count": int(np.sum(mask)),
                        "unique_pairs": len(groups),
                        "mean_delta": float(np.mean(delta_values)),
                        "mean_delta_ci_low": delta_ci[0],
                        "mean_delta_ci_high": delta_ci[1],
                        "win_rate": float(np.mean(win_values)),
                        "win_rate_ci_low": win_ci[0],
                        "win_rate_ci_high": win_ci[1],
                        "win_rate_wilson_ci_low": wilson_ci[0],
                        "win_rate_wilson_ci_high": wilson_ci[1],
                    }
                )
    return rows


def comparison_rows(
    data: ModelData,
    replicates: int,
    seed: int,
) -> list[dict[str, Any]]:
    masks: list[tuple[str, str, str, np.ndarray]] = [
        ("all", "all", "all", np.ones(len(data.delta), dtype=bool)),
        ("strict", "all", "add+substitute", data.operation != "delete"),
    ]
    for operation in EDIT_OPERATIONS:
        masks.append((operation, "all", operation, data.operation == operation))
    for semantic in SEMANTIC_TARGETS:
        for operation in EDIT_OPERATIONS:
            masks.append(
                (
                    f"{semantic}|{operation}",
                    semantic,
                    operation,
                    (data.semantic == semantic) & (data.operation == operation),
                )
            )

    rows = []
    for index, (scope, semantic, operation, mask) in enumerate(masks):
        rows.append(
            pair_balanced_comparison(
                data,
                mask,
                scope,
                semantic,
                operation,
                replicates,
                seed + index,
            )
        )
        rows.append(
            matched_pair_cell_comparison(
                data,
                mask,
                scope,
                semantic,
                operation,
                replicates,
                seed + 2000 + index,
            )
        )
    rows.append(
        cell_macro_comparison(
            data,
            np.ones(len(data.delta), dtype=bool),
            CELL_LABELS,
            "all",
            replicates,
            seed + 4000,
        )
    )
    strict_cells = tuple(
        cell for cell in CELL_LABELS if not cell.endswith("|delete")
    )
    rows.append(
        cell_macro_comparison(
            data,
            data.operation != "delete",
            strict_cells,
            "strict",
            replicates,
            seed + 5000,
        )
    )
    return rows


def empirical_zoom(data_by_model: Sequence[ModelData]) -> tuple[float, float]:
    minimum = min(float(data.delta.min()) for data in data_by_model)
    maximum = max(float(data.delta.max()) for data in data_by_model)
    lower = math.floor(minimum / 0.05) * 0.05
    upper = math.ceil(maximum / 0.05) * 0.05
    return lower, upper


def save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}{path.suffix}")
    fig.savefig(temporary, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    temporary.replace(path)


def plot_model_mode_histograms(
    data_by_model: Sequence[ModelData],
    output_dir: Path,
    x_range: tuple[float, float],
    bins: int,
    suffix: str,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(
        len(data_by_model),
        len(CONSTRUCTION_MODES),
        figsize=(13, 9),
        sharex=True,
        sharey="row",
        constrained_layout=True,
    )
    edges = np.linspace(x_range[0], x_range[1], bins + 1)
    for row_index, data in enumerate(data_by_model):
        for column_index, mode in enumerate(CONSTRUCTION_MODES):
            axis = axes[row_index, column_index]
            values = data.delta[data.mode == mode]
            axis.hist(
                values,
                bins=edges,
                color=MODE_COLORS[mode],
                alpha=0.85,
                edgecolor="white",
                linewidth=0.3,
            )
            axis.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
            axis.set_xlim(*x_range)
            axis.grid(axis="y", alpha=0.2)
            axis.set_title(
                f"{MODEL_SHORT[data.name]} — {MODE_LABELS[mode]}\n"
                f"n={len(values):,}, win={np.mean(values > 0):.1%}",
                fontsize=10,
            )
            if column_index == 0:
                axis.set_ylabel("Sample count")
            if row_index == len(data_by_model) - 1:
                axis.set_xlabel("Delta = sim(q+, t) - sim(q-, t)")
    title_suffix = (
        "theoretical cosine-difference range [-2, 2]"
        if x_range == (-2.0, 2.0)
        else f"empirical zoom [{x_range[0]:.2f}, {x_range[1]:.2f}]"
    )
    fig.suptitle(f"Delta distributions by model and construction mode\n{title_suffix}")
    save_figure(fig, output_dir / f"delta_hist_by_model_mode_{suffix}.png", dpi)


def plot_cell_histograms(
    data: ModelData,
    output_dir: Path,
    x_range: tuple[float, float],
    dpi: int,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True, constrained_layout=True)
    edges = np.linspace(x_range[0], x_range[1], 46)
    for row_index, semantic in enumerate(SEMANTIC_TARGETS):
        for column_index, operation in enumerate(EDIT_OPERATIONS):
            axis = axes[row_index, column_index]
            for mode in CONSTRUCTION_MODES:
                mask = (
                    (data.semantic == semantic)
                    & (data.operation == operation)
                    & (data.mode == mode)
                )
                values = data.delta[mask]
                axis.hist(
                    values,
                    bins=edges,
                    histtype="step",
                    linewidth=1.8,
                    color=MODE_COLORS[mode],
                    label=(
                        f"{MODE_LABELS[mode]} n={len(values)}, "
                        f"win={np.mean(values > 0):.1%}"
                    ),
                )
            axis.axvline(0.0, color="black", linestyle="--", linewidth=0.9)
            axis.set_xlim(*x_range)
            axis.grid(axis="y", alpha=0.2)
            axis.set_title(f"{semantic} × {operation}")
            axis.legend(fontsize=7, frameon=False)
            if column_index == 0:
                axis.set_ylabel("Sample count")
            if row_index == 2:
                axis.set_xlabel("Delta (empirical zoom)")
    fig.suptitle(
        f"{MODEL_SHORT[data.name]}: Delta count histograms by 3×3 cell and mode\n"
        "All panels share bins; theoretical Delta range is [-2, 2]"
    )
    slug = data.name.replace("/", "_")
    save_figure(fig, output_dir / f"delta_hist_3x3_{slug}.png", dpi)


def plot_cell_win_rates(
    data_by_model: Sequence[ModelData],
    win_rows: Sequence[dict[str, Any]],
    output_dir: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True, constrained_layout=True)
    x = np.arange(len(CELL_LABELS))
    width = 0.38
    for axis, data in zip(axes, data_by_model):
        model_rows = [row for row in win_rows if row["model"] == data.name]
        for mode_index, mode in enumerate(CONSTRUCTION_MODES):
            values = []
            low = []
            high = []
            for cell in CELL_LABELS:
                semantic, operation = cell.split("|")
                row = next(
                    item
                    for item in model_rows
                    if item["construction_mode"] == mode
                    and item["semantic_target"] == semantic
                    and item["edit_operation"] == operation
                )
                value = float(row["win_rate"])
                values.append(value)
                low.append(value - float(row["win_rate_wilson_ci_low"]))
                high.append(float(row["win_rate_wilson_ci_high"]) - value)
            positions = x + (mode_index - 0.5) * width
            axis.bar(
                positions,
                values,
                width=width,
                color=MODE_COLORS[mode],
                alpha=0.85,
                label=MODE_LABELS[mode],
                yerr=np.asarray([low, high]),
                error_kw={"elinewidth": 0.8, "capsize": 2},
            )
        axis.axhline(0.5, color="black", linestyle="--", linewidth=1.0)
        axis.set_ylim(0.0, 1.05)
        axis.set_ylabel("Win rate (Delta > 0)")
        axis.set_title(MODEL_SHORT[data.name])
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False, loc="lower right")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(
        [cell.replace("object", "obj").replace("attribute", "attr").replace("predicate", "pred").replace("|", "\n") for cell in CELL_LABELS]
    )
    fig.suptitle(
        "Delta win rate by model, 3×3 cell, and construction mode\n"
        "95% Wilson CI (conservative for sparse all-win/all-loss cells)"
    )
    save_figure(fig, output_dir / "delta_win_rate_3x3_by_mode.png", dpi)


def plot_mode_summary(
    data_by_model: Sequence[ModelData],
    comparison_rows_value: Sequence[dict[str, Any]],
    output_dir: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    x = np.arange(len(data_by_model))
    width = 0.36
    for axis, scope in zip(axes, ("all", "strict")):
        for mode_index, mode in enumerate(CONSTRUCTION_MODES):
            values = []
            for data in data_by_model:
                row = next(
                    item
                    for item in comparison_rows_value
                    if item["model"] == data.name
                    and item["comparison"] == "pair_balanced"
                    and item["scope"] == scope
                )
                values.append(
                    float(
                        row[
                            "text_win_rate"
                            if mode == "text_only_editable"
                            else "vision_win_rate"
                        ]
                    )
                )
            axis.bar(
                x + (mode_index - 0.5) * width,
                values,
                width=width,
                color=MODE_COLORS[mode],
                label=MODE_LABELS[mode],
                alpha=0.85,
            )
        axis.axhline(0.5, color="black", linestyle="--", linewidth=1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_xticks(x)
        axis.set_xticklabels([MODEL_SHORT[data.name] for data in data_by_model])
        axis.set_ylabel("Pair-balanced win rate")
        axis.set_title("All 9 cells" if scope == "all" else "Strict: Add + Substitute")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False)
    fig.suptitle("Text-only vs vision-grounded Delta win rates")
    save_figure(fig, output_dir / "delta_win_rate_mode_summary.png", dpi)


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates <= 0:
        raise ValueError("--bootstrap-replicates must be positive")
    diagnostics_root = args.diagnostics_root.resolve()
    output_dir = (args.output_dir or diagnostics_root / "delta_mode_analysis").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data_by_model = [load_model_data(diagnostics_root / model) for model in MODEL_ORDER]
    canonical = data_by_model[0]
    for data in data_by_model[1:]:
        for field in ("mode", "semantic", "operation", "pair", "cell"):
            if not np.array_equal(getattr(canonical, field), getattr(data, field)):
                raise ValueError(f"Row labels differ between models for field {field}")

    win_rows: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for model_index, data in enumerate(data_by_model):
        win_rows.extend(
            mode_win_rate_rows(
                data,
                args.bootstrap_replicates,
                args.seed + model_index * 10000,
            )
        )
        comparisons.extend(
            comparison_rows(
                data,
                args.bootstrap_replicates,
                args.seed + model_index * 10000,
            )
        )

    win_fields = [
        "model",
        "construction_mode",
        "semantic_target",
        "edit_operation",
        "count",
        "unique_pairs",
        "mean_delta",
        "mean_delta_ci_low",
        "mean_delta_ci_high",
        "win_rate",
        "win_rate_ci_low",
        "win_rate_ci_high",
        "win_rate_wilson_ci_low",
        "win_rate_wilson_ci_high",
    ]
    comparison_fields = [
        "model",
        "comparison",
        "scope",
        "semantic_target",
        "edit_operation",
        "text_rows",
        "vision_rows",
        "text_pairs",
        "vision_pairs",
        "matched_pair_cells",
        "matched_pairs",
        "text_win_rate",
        "vision_win_rate",
        "win_rate_diff_vision_minus_text",
        "win_rate_diff_ci_low",
        "win_rate_diff_ci_high",
        "text_mean_delta",
        "vision_mean_delta",
        "mean_delta_diff_vision_minus_text",
        "mean_delta_diff_ci_low",
        "mean_delta_diff_ci_high",
    ]
    write_csv_atomic(output_dir / "mode_win_rates_by_cell.csv", win_rows, win_fields)
    write_csv_atomic(output_dir / "mode_comparisons.csv", comparisons, comparison_fields)

    zoom = empirical_zoom(data_by_model)
    plot_model_mode_histograms(
        data_by_model,
        output_dir,
        (-2.0, 2.0),
        bins=160,
        suffix="full_range",
        dpi=args.dpi,
    )
    plot_model_mode_histograms(
        data_by_model,
        output_dir,
        zoom,
        bins=60,
        suffix="zoom",
        dpi=args.dpi,
    )
    for data in data_by_model:
        plot_cell_histograms(data, output_dir, zoom, args.dpi)
    plot_cell_win_rates(data_by_model, win_rows, output_dir, args.dpi)
    plot_mode_summary(data_by_model, comparisons, output_dir, args.dpi)

    metadata = {
        "schema_version": "1.0",
        "theoretical_delta_range": [-2.0, 2.0],
        "empirical_plot_range": [zoom[0], zoom[1]],
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "models": {
            data.name: {
                "count": len(data.delta),
                "minimum": float(data.delta.min()),
                "maximum": float(data.delta.max()),
                "quantiles": {
                    str(quantile): float(np.quantile(data.delta, quantile))
                    for quantile in (0.001, 0.01, 0.5, 0.99, 0.999)
                },
            }
            for data in data_by_model
        },
        "files": {
            "mode_win_rates_by_cell": "mode_win_rates_by_cell.csv",
            "mode_comparisons": "mode_comparisons.csv",
            "full_range_histogram": "delta_hist_by_model_mode_full_range.png",
            "zoom_histogram": "delta_hist_by_model_mode_zoom.png",
            "cell_win_rates": "delta_win_rate_3x3_by_mode.png",
            "mode_summary": "delta_win_rate_mode_summary.png",
            "cell_histograms": [
                f"delta_hist_3x3_{model}.png" for model in MODEL_ORDER
            ],
        },
    }
    write_json_atomic(output_dir / "metadata.json", metadata)
    print(f"[done] output={output_dir} zoom={zoom} comparisons={len(comparisons)}")


if __name__ == "__main__":
    main()
