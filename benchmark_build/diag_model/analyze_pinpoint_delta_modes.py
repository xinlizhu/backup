#!/usr/bin/env python3
"""Offline Delta distribution analysis for PinPoint V7 diagnostics.

PinPoint V7 has one construction protocol (text-only generation followed by
vision-grounded selection), so this script analyzes actual V7 factors instead
of pretending that the old text-only/vision-grounded construction modes exist.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cir-qdiag-matplotlib")

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import (  # noqa: E402
    CELL_LABELS,
    EDIT_OPERATIONS,
    SEMANTIC_TARGETS,
    write_json_atomic,
)


MODEL_ORDER = ("BGE-VL-base", "BGE-VL-MLLM-S1")
MODEL_SHORT = {
    "BGE-VL-base": "BGE-base",
    "BGE-VL-MLLM-S1": "MLLM-S1",
}
MODEL_COLORS = {
    "BGE-VL-base": "#4C78A8",
    "BGE-VL-MLLM-S1": "#F58518",
}
OPERATION_COLORS = {
    "add": "#2A9D8F",
    "delete": "#D45087",
    "substitute": "#B07919",
}
REVIEW_COLORS = {
    "fully_reviewed": "#59A14F",
    "partially_reviewed": "#E5AE38",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze PinPoint V7 Delta distributions by operation and 3x3 cell."
    )
    parser.add_argument(
        "--diagnostics-root",
        required=True,
        type=Path,
        help="PinPoint V7 full_retrieval_diagnostics directory.",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="PinPoint V7 final.jsonl, used for review-scope labels.",
    )
    parser.add_argument(
        "--models",
        default=",".join(MODEL_ORDER),
        help=f"Comma-separated subset of {','.join(MODEL_ORDER)}.",
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
    parser.add_argument("--zoom-bin-width", type=float, default=0.01)
    return parser.parse_args()


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


def load_review_scope(input_path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            group = json.loads(line)
            group_id = str(group["group_id"])
            positive_count = len(group.get("positive_images", []))
            reviewed_count = len(
                group.get("metadata", {}).get("validated_positive_image_ids", [])
            )
            result[group_id] = (
                "fully_reviewed"
                if positive_count == reviewed_count
                else "partially_reviewed"
            )
    if not result:
        raise ValueError(f"{input_path} contains no PinPoint groups")
    return result


def load_model_frame(
    diagnostics_root: Path,
    model_name: str,
    review_scope: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    analysis_dir = diagnostics_root / model_name / "analysis"
    metrics_path = analysis_dir / "negative_pair_metrics.csv"
    calibration_path = analysis_dir / "calibration.json"
    metadata_path = diagnostics_root / model_name / "metadata.json"
    for path in (metrics_path, calibration_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing PinPoint V7 diagnostic output: {path}")
    frame = pd.read_csv(metrics_path)
    required = {
        "group_id",
        "instruction_id",
        "negative_id",
        "semantic_target",
        "edit_operation",
        "mean_positive_query_target_score",
        "mean_negative_query_target_score",
        "mean_target_delta",
        "target_raw_win_rate",
        "target_robust_win_rate",
        "delta_ndcg@10",
        "source_negative_top10_jaccard",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"{metrics_path} is missing columns: {sorted(missing)}")
    frame["model"] = model_name
    frame["cell"] = (
        frame["semantic_target"].astype(str)
        + "|"
        + frame["edit_operation"].astype(str)
    )
    frame["review_scope"] = frame["group_id"].map(review_scope)
    if frame["review_scope"].isna().any():
        missing_groups = sorted(
            frame.loc[frame["review_scope"].isna(), "group_id"].unique().tolist()
        )
        raise ValueError(f"Missing review scope for groups: {missing_groups[:5]}")
    delta = frame["mean_target_delta"].to_numpy(dtype=np.float64)
    if delta.min(initial=0.0) < -2.00001 or delta.max(initial=0.0) > 2.00001:
        raise AssertionError("Mean cosine Delta must lie in [-2, 2]")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return frame, {
        "normal_perturbation_threshold": float(
            calibration["normal_perturbation_threshold"]
        ),
        "threshold_bootstrap_ci": calibration["threshold_bootstrap_ci"],
        "query_fingerprint": metadata["query_fingerprint"],
        "input_sha256": metadata["input_sha256"],
    }


def hierarchical_group_values(frame: pd.DataFrame, column: str) -> pd.Series:
    return (
        frame.groupby(["group_id", "instruction_id"], sort=False)[column]
        .mean()
        .groupby("group_id", sort=False)
        .mean()
    )


def bootstrap_mean(
    values: np.ndarray,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if not len(values):
        return float("nan"), float("nan")
    if replicates <= 0:
        point = float(np.mean(values))
        return point, point
    samples = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 256):
        end = min(start + 256, replicates)
        chosen = rng.integers(0, len(values), size=(end - start, len(values)))
        samples[start:end] = np.mean(values[chosen], axis=1)
    return (
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    )


def subset_specs(frame: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    specs: list[tuple[str, pd.Series]] = [
        ("all", pd.Series(True, index=frame.index)),
        ("strict", frame["edit_operation"].isin(["add", "substitute"])),
    ]
    for operation in EDIT_OPERATIONS:
        specs.append((operation, frame["edit_operation"].eq(operation)))
    for cell in CELL_LABELS:
        specs.append((cell, frame["cell"].eq(cell)))
    for scope in ("fully_reviewed", "partially_reviewed"):
        specs.append((f"strict|{scope}", (
            frame["edit_operation"].isin(["add", "substitute"])
            & frame["review_scope"].eq(scope)
        )))
    return specs


def summarize_model(
    frame: pd.DataFrame,
    model_name: str,
    replicates: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metric_columns = (
        "mean_positive_query_target_score",
        "mean_negative_query_target_score",
        "mean_target_delta",
        "target_raw_win_rate",
        "target_robust_win_rate",
        "delta_ndcg@10",
        "source_negative_top10_jaccard",
    )
    quantiles = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
    for subset_index, (subset_name, mask) in enumerate(subset_specs(frame)):
        subset = frame.loc[mask].copy()
        if subset.empty:
            continue
        row: dict[str, Any] = {
            "model": model_name,
            "subset": subset_name,
            "negative_count": len(subset),
            "instruction_count": subset[
                ["group_id", "instruction_id"]
            ].drop_duplicates().shape[0],
            "group_count": subset["group_id"].nunique(),
        }
        for metric_index, column in enumerate(metric_columns):
            values = hierarchical_group_values(subset, column).to_numpy(
                dtype=np.float64
            )
            row[f"pair_balanced_{column}"] = float(np.mean(values))
            low, high = bootstrap_mean(
                values,
                replicates,
                np.random.default_rng(
                    seed + subset_index * 100 + metric_index
                ),
            )
            row[f"{column}_ci_low"] = low
            row[f"{column}_ci_high"] = high
        delta = subset["mean_target_delta"].to_numpy(dtype=np.float64)
        for quantile in quantiles:
            row[f"row_delta_q{int(100 * quantile):02d}"] = float(
                np.quantile(delta, quantile)
            )
        rows.append(row)
    return rows


def empirical_zoom(frames: dict[str, pd.DataFrame]) -> tuple[float, float]:
    minimum = min(float(frame["mean_target_delta"].min()) for frame in frames.values())
    maximum = max(float(frame["mean_target_delta"].max()) for frame in frames.values())
    lower = min(-0.05, math.floor(minimum / 0.05) * 0.05)
    upper = max(0.05, math.ceil(maximum / 0.05) * 0.05)
    return lower, upper


def save_figure(figure: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}{path.suffix}")
    figure.savefig(temporary, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    temporary.replace(path)


def plot_model_operation_histograms(
    frames: dict[str, pd.DataFrame],
    thresholds: dict[str, float],
    models: Sequence[str],
    output_dir: Path,
    x_range: tuple[float, float],
    bins: int,
    suffix: str,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(
        len(models),
        len(EDIT_OPERATIONS),
        figsize=(15, 4.2 * len(models)),
        sharex=True,
        sharey="row",
        constrained_layout=True,
        squeeze=False,
    )
    edges = np.linspace(x_range[0], x_range[1], bins + 1)
    for row_index, model_name in enumerate(models):
        frame = frames[model_name]
        threshold = thresholds[model_name]
        for column_index, operation in enumerate(EDIT_OPERATIONS):
            axis = axes[row_index, column_index]
            values = frame.loc[
                frame["edit_operation"].eq(operation),
                "mean_target_delta",
            ].to_numpy(dtype=np.float64)
            axis.hist(
                values,
                bins=edges,
                color=OPERATION_COLORS[operation],
                alpha=0.86,
                edgecolor="white",
                linewidth=0.25,
            )
            axis.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
            axis.axvline(
                threshold,
                color="#C1121F",
                linestyle=":",
                linewidth=1.2,
                label=f"τ={threshold:.3f}",
            )
            axis.set_xlim(*x_range)
            axis.grid(axis="y", alpha=0.2)
            axis.set_title(
                f"{MODEL_SHORT[model_name]} — {operation.upper()}\n"
                f"n={len(values):,}, row win={np.mean(values > 0):.1%}",
                fontsize=10,
            )
            axis.legend(fontsize=8)
            if column_index == 0:
                axis.set_ylabel("Accepted-negative count")
            if row_index == len(models) - 1:
                axis.set_xlabel("Mean target Delta = mean_t[sim(q+,t)-sim(q-,t)]")
    range_text = (
        "theoretical cosine-difference range [-2, 2]"
        if x_range == (-2.0, 2.0)
        else f"empirical zoom [{x_range[0]:.2f}, {x_range[1]:.2f}]"
    )
    figure.suptitle(
        f"PinPoint V7 mean-target Delta by model and operation\n{range_text}",
        fontsize=15,
    )
    save_figure(
        figure,
        output_dir / f"delta_hist_by_model_operation_{suffix}.png",
        dpi,
    )


def plot_cell_histograms(
    frame: pd.DataFrame,
    model_name: str,
    threshold: float,
    output_dir: Path,
    x_range: tuple[float, float],
    dpi: int,
) -> None:
    figure, axes = plt.subplots(
        3,
        3,
        figsize=(15, 11),
        sharex=True,
        constrained_layout=True,
    )
    edges = np.linspace(x_range[0], x_range[1], 51)
    for row_index, semantic in enumerate(SEMANTIC_TARGETS):
        for column_index, operation in enumerate(EDIT_OPERATIONS):
            axis = axes[row_index, column_index]
            selected = frame[
                frame["semantic_target"].eq(semantic)
                & frame["edit_operation"].eq(operation)
            ]
            values = selected["mean_target_delta"].to_numpy(dtype=np.float64)
            axis.hist(
                values,
                bins=edges,
                color=OPERATION_COLORS[operation],
                alpha=0.84,
                edgecolor="white",
                linewidth=0.25,
            )
            axis.axvline(0.0, color="black", linestyle="--", linewidth=0.9)
            axis.axvline(
                threshold,
                color="#C1121F",
                linestyle=":",
                linewidth=1.0,
            )
            axis.set_title(
                f"{semantic} × {operation}\n"
                f"n={len(values):,}, row win={np.mean(values > 0):.1%}",
                fontsize=10,
            )
            axis.grid(axis="y", alpha=0.2)
            if column_index == 0:
                axis.set_ylabel("Count")
            if row_index == 2:
                axis.set_xlabel("Mean target Delta")
    figure.suptitle(
        f"PinPoint V7 3×3 mean-target Delta — {MODEL_SHORT[model_name]}\n"
        f"zoom [{x_range[0]:.2f}, {x_range[1]:.2f}], "
        f"red dotted line is τ={threshold:.3f}",
        fontsize=15,
    )
    save_figure(
        figure,
        output_dir / f"delta_hist_3x3_{model_name}.png",
        dpi,
    )


def plot_win_rate_cells(
    summary: pd.DataFrame,
    models: Sequence[str],
    output_dir: Path,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(
        3,
        3,
        figsize=(16, 11),
        sharey=True,
        constrained_layout=True,
    )
    for row_index, semantic in enumerate(SEMANTIC_TARGETS):
        for column_index, operation in enumerate(EDIT_OPERATIONS):
            axis = axes[row_index, column_index]
            subset_name = f"{semantic}|{operation}"
            positions: list[float] = []
            heights: list[float] = []
            errors_low: list[float] = []
            errors_high: list[float] = []
            colors: list[str] = []
            labels: list[str] = []
            for model_index, model_name in enumerate(models):
                row = summary[
                    summary["model"].eq(model_name)
                    & summary["subset"].eq(subset_name)
                ].iloc[0]
                for metric_index, (metric, short, alpha) in enumerate(
                    (
                        ("target_raw_win_rate", "raw", 0.95),
                        ("target_robust_win_rate", "rob", 0.48),
                    )
                ):
                    value = float(row[f"pair_balanced_{metric}"])
                    low = float(row[f"{metric}_ci_low"])
                    high = float(row[f"{metric}_ci_high"])
                    positions.append(model_index * 3 + metric_index)
                    heights.append(value)
                    errors_low.append(value - low)
                    errors_high.append(high - value)
                    base_color = MODEL_COLORS[model_name]
                    colors.append(base_color)
                    labels.append(f"{MODEL_SHORT[model_name]}\n{short}")
                    axis.bar(
                        positions[-1],
                        value,
                        width=0.82,
                        color=base_color,
                        alpha=alpha,
                    )
            axis.errorbar(
                positions,
                heights,
                yerr=np.asarray([errors_low, errors_high]),
                fmt="none",
                ecolor="black",
                elinewidth=0.8,
                capsize=2,
            )
            axis.set_xticks(positions, labels, fontsize=7)
            axis.set_ylim(0.0, 1.0)
            axis.set_title(f"{semantic} × {operation}")
            axis.grid(axis="y", alpha=0.2)
            if column_index == 0:
                axis.set_ylabel("Pair-balanced target win rate")
    figure.suptitle(
        "PinPoint V7 3×3 raw and Q95-robust target win rates\n"
        "robust bars use each model's positive-paraphrase threshold",
        fontsize=15,
    )
    save_figure(figure, output_dir / "delta_win_rate_3x3.png", dpi)


def plot_model_win_rate_heatmap(
    summary: pd.DataFrame,
    model_name: str,
    output_dir: Path,
    raw_range: tuple[float, float],
    robust_range: tuple[float, float],
    dpi: int,
) -> None:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5.7),
        constrained_layout=True,
    )
    metric_specs = (
        (
            "target_raw_win_rate",
            r"Raw target win: $\Delta_t > 0$",
            "Blues",
            raw_range,
        ),
        (
            "target_robust_win_rate",
            r"Q95-robust target win: $\Delta_t > \tau$",
            "Oranges",
            robust_range,
        ),
    )
    for axis, (metric, title, color_map, value_range) in zip(
        axes,
        metric_specs,
        strict=True,
    ):
        values = np.zeros((len(SEMANTIC_TARGETS), len(EDIT_OPERATIONS)))
        lows = np.zeros_like(values)
        highs = np.zeros_like(values)
        for row_index, semantic in enumerate(SEMANTIC_TARGETS):
            for column_index, operation in enumerate(EDIT_OPERATIONS):
                row = summary[
                    summary["model"].eq(model_name)
                    & summary["subset"].eq(f"{semantic}|{operation}")
                ].iloc[0]
                values[row_index, column_index] = row[f"pair_balanced_{metric}"]
                lows[row_index, column_index] = row[f"{metric}_ci_low"]
                highs[row_index, column_index] = row[f"{metric}_ci_high"]

        image = axis.imshow(
            values,
            cmap=color_map,
            vmin=value_range[0],
            vmax=value_range[1],
            aspect="equal",
        )
        axis.set_xticks(
            np.arange(len(EDIT_OPERATIONS)),
            [operation.upper() for operation in EDIT_OPERATIONS],
        )
        axis.set_yticks(
            np.arange(len(SEMANTIC_TARGETS)),
            [semantic.title() for semantic in SEMANTIC_TARGETS],
        )
        axis.set_xlabel("Edit operation")
        axis.set_ylabel("Semantic target")
        axis.set_title(title)
        normalized = np.clip(
            (values - value_range[0]) / max(value_range[1] - value_range[0], 1e-12),
            0.0,
            1.0,
        )
        for row_index in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                text_color = "white" if normalized[row_index, column_index] > 0.56 else "black"
                axis.text(
                    column_index,
                    row_index,
                    f"{values[row_index, column_index]:.1%}\n"
                    f"[{lows[row_index, column_index]:.1%}, "
                    f"{highs[row_index, column_index]:.1%}]",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=10,
                    fontweight="semibold",
                )
        colorbar = figure.colorbar(image, ax=axis, shrink=0.86, pad=0.03)
        colorbar.set_label("Pair-balanced target win rate")
        colorbar.ax.yaxis.set_major_formatter(
            matplotlib.ticker.PercentFormatter(xmax=1.0)
        )

    figure.suptitle(
        f"PinPoint V7 3×3 Delta target win-rate heatmaps — "
        f"{MODEL_SHORT[model_name]}\n"
        "cell text: pair-balanced estimate [group-bootstrap 95% CI]",
        fontsize=15,
    )
    save_figure(
        figure,
        output_dir / f"delta_win_rate_heatmap_{model_name}.png",
        dpi,
    )


def plot_strict_ecdf(
    frames: dict[str, pd.DataFrame],
    thresholds: dict[str, float],
    models: Sequence[str],
    output_dir: Path,
    x_range: tuple[float, float],
    dpi: int,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for model_name in models:
        frame = frames[model_name]
        values = np.sort(
            frame.loc[
                frame["edit_operation"].isin(["add", "substitute"]),
                "mean_target_delta",
            ].to_numpy(dtype=np.float64)
        )
        probabilities = np.arange(1, len(values) + 1) / len(values)
        color = MODEL_COLORS[model_name]
        axis.plot(
            values,
            probabilities,
            color=color,
            linewidth=1.8,
            label=f"{MODEL_SHORT[model_name]} strict (n={len(values):,})",
        )
        axis.axvline(
            thresholds[model_name],
            color=color,
            linestyle=":",
            linewidth=1.2,
            label=f"{MODEL_SHORT[model_name]} τ={thresholds[model_name]:.3f}",
        )
    axis.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    axis.set_xlim(*x_range)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Mean target Delta")
    axis.set_ylabel("Empirical cumulative probability")
    axis.set_title("PinPoint V7 strict mean-target Delta ECDF")
    axis.grid(alpha=0.2)
    axis.legend()
    save_figure(figure, output_dir / "delta_ecdf_strict.png", dpi)


def plot_review_scope(
    summary: pd.DataFrame,
    models: Sequence[str],
    output_dir: Path,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for model_index, model_name in enumerate(models):
        for scope_index, scope in enumerate(("fully_reviewed", "partially_reviewed")):
            row = summary[
                summary["model"].eq(model_name)
                & summary["subset"].eq(f"strict|{scope}")
            ].iloc[0]
            position = model_index * 3 + scope_index
            axes[0].bar(
                position,
                row["pair_balanced_target_robust_win_rate"],
                color=REVIEW_COLORS[scope],
                label=scope if model_index == 0 else None,
            )
            axes[1].bar(
                position,
                row["pair_balanced_delta_ndcg@10"],
                color=REVIEW_COLORS[scope],
                label=scope if model_index == 0 else None,
            )
    tick_positions = [0.5, 3.5]
    tick_labels = [MODEL_SHORT[model_name] for model_name in models]
    for axis, title, ylabel in (
        (axes[0], "Q95-robust target win rate", "Rate"),
        (axes[1], "Delta NDCG@10", "Positive − residual NDCG"),
    ):
        axis.set_xticks(tick_positions, tick_labels)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle(
        "PinPoint V7 strict diagnostics by Positive-image review scope\n"
        "descriptive comparison; group content is not matched",
        fontsize=14,
    )
    save_figure(figure, output_dir / "delta_by_review_scope.png", dpi)


def histogram_rows(
    frames: dict[str, pd.DataFrame],
    models: Sequence[str],
    edges: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scopes = {
        "all": lambda frame: pd.Series(True, index=frame.index),
        "strict": lambda frame: frame["edit_operation"].isin(["add", "substitute"]),
        "add": lambda frame: frame["edit_operation"].eq("add"),
        "delete": lambda frame: frame["edit_operation"].eq("delete"),
        "substitute": lambda frame: frame["edit_operation"].eq("substitute"),
    }
    for model_name in models:
        frame = frames[model_name]
        for scope_name, mask_fn in scopes.items():
            values = frame.loc[mask_fn(frame), "mean_target_delta"].to_numpy(
                dtype=np.float64
            )
            counts, _ = np.histogram(values, bins=edges)
            for index, count in enumerate(counts):
                rows.append(
                    {
                        "model": model_name,
                        "scope": scope_name,
                        "bin_left": float(edges[index]),
                        "bin_right": float(edges[index + 1]),
                        "count": int(count),
                    }
                )
    return rows


def make_summary(
    summary: pd.DataFrame,
    thresholds: dict[str, float],
    models: Sequence[str],
    zoom: tuple[float, float],
) -> str:
    lines = [
        "# PinPoint V7 Delta mode analysis",
        "",
        "PinPoint V7 does not contain the old benchmark's paired "
        "`text_only_editable/vision_grounded` construction modes. Every V7 candidate "
        "is text-only generated and then vision-grounded selected. This directory keeps "
        "the historical name `delta_mode_analysis`, but analyzes model, edit operation, "
        "3×3 cell, and Positive-review scope.",
        "",
        "The histogram variable is each accepted negative's mean target Delta:",
        "",
        "```text",
        "mean_t[sim(f(R,m+),t) - sim(f(R,m-),t)]",
        "```",
        "",
        f"Empirical zoom used by plots: `[{zoom[0]:.2f}, {zoom[1]:.2f}]`.",
        "",
        "## Strict overview",
        "",
        "| Model | tau | q+ score | q- score | mean Delta | raw target win | "
        "robust target win | Delta NDCG@10 | Jaccard@10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in models:
        row = summary[
            summary["model"].eq(model_name) & summary["subset"].eq("strict")
        ].iloc[0]
        lines.append(
            "| "
            + " | ".join(
                [
                    model_name,
                    f"{thresholds[model_name]:.4f}",
                    f"{row['pair_balanced_mean_positive_query_target_score']:.4f}",
                    f"{row['pair_balanced_mean_negative_query_target_score']:.4f}",
                    f"{row['pair_balanced_mean_target_delta']:.4f}",
                    f"{row['pair_balanced_target_raw_win_rate']:.2%}",
                    f"{row['pair_balanced_target_robust_win_rate']:.2%}",
                    f"{row['pair_balanced_delta_ndcg@10']:.4f}",
                    f"{row['pair_balanced_source_negative_top10_jaccard']:.4f}",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Win-rate heatmaps", ""])
    for model_name in models:
        lines.append(
            f"- [{MODEL_SHORT[model_name]} raw/robust 3×3 heatmaps]"
            f"(delta_win_rate_heatmap_{model_name}.png)"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- Full-range histograms use the theoretical cosine-difference interval "
            "`[-2,2]`; zoom plots show the actual V7 concentration.",
            "- Histogram counts are accepted-negative rows. Summary means and confidence "
            "intervals are hierarchical group-balanced statistics.",
            "- The red threshold line is model-specific positive-paraphrase Q95.",
            "- DELETE is a weaker-query diagnostic and should not be interpreted as a "
            "strict hard-negative success rate.",
            "- Review-scope plots are descriptive because fully and partially reviewed "
            "groups are not content matched.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    diagnostics_root = args.diagnostics_root.resolve()
    input_path = args.input.resolve()
    output_dir = (
        args.output_dir or diagnostics_root / "delta_mode_analysis"
    ).resolve()
    models = [value.strip() for value in args.models.split(",") if value.strip()]
    unknown = sorted(set(models) - set(MODEL_ORDER))
    if unknown:
        raise ValueError(f"Unsupported models: {unknown}")
    if not models:
        raise ValueError("--models selected no models")
    if args.bootstrap_replicates < 0:
        raise ValueError("--bootstrap-replicates must be non-negative")
    if args.zoom_bin_width <= 0:
        raise ValueError("--zoom-bin-width must be positive")

    review_scope = load_review_scope(input_path)
    frames: dict[str, pd.DataFrame] = {}
    model_metadata: dict[str, dict[str, Any]] = {}
    for model_name in models:
        frames[model_name], model_metadata[model_name] = load_model_frame(
            diagnostics_root,
            model_name,
            review_scope,
        )
    first_ids = frames[models[0]]["negative_id"].astype(str).tolist()
    for model_name in models[1:]:
        if frames[model_name]["negative_id"].astype(str).tolist() != first_ids:
            raise ValueError("Models do not contain identical ordered V7 negative IDs")

    thresholds = {
        model_name: model_metadata[model_name]["normal_perturbation_threshold"]
        for model_name in models
    }
    summary_rows: list[dict[str, Any]] = []
    for model_index, model_name in enumerate(models):
        summary_rows.extend(
            summarize_model(
                frames[model_name],
                model_name,
                args.bootstrap_replicates,
                args.seed + model_index * 10000,
            )
        )
    summary_frame = pd.DataFrame(summary_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "delta_summary.csv"
    temporary_summary = summary_path.with_name(
        f".{summary_path.name}.{os.getpid()}.tmp"
    )
    summary_frame.to_csv(temporary_summary, index=False)
    temporary_summary.replace(summary_path)

    zoom = empirical_zoom(frames)
    zoom_bins = max(10, int(math.ceil((zoom[1] - zoom[0]) / args.zoom_bin_width)))
    plot_model_operation_histograms(
        frames,
        thresholds,
        models,
        output_dir,
        (-2.0, 2.0),
        100,
        "full_range",
        args.dpi,
    )
    plot_model_operation_histograms(
        frames,
        thresholds,
        models,
        output_dir,
        zoom,
        zoom_bins,
        "zoom",
        args.dpi,
    )
    for model_name in models:
        plot_cell_histograms(
            frames[model_name],
            model_name,
            thresholds[model_name],
            output_dir,
            zoom,
            args.dpi,
        )
    plot_win_rate_cells(summary_frame, models, output_dir, args.dpi)
    cell_summary = summary_frame[summary_frame["subset"].str.contains(r"\|")]
    raw_values = cell_summary["pair_balanced_target_raw_win_rate"].to_numpy(
        dtype=np.float64
    )
    robust_values = cell_summary[
        "pair_balanced_target_robust_win_rate"
    ].to_numpy(dtype=np.float64)
    raw_range = (
        max(0.0, math.floor((float(raw_values.min()) - 0.05) / 0.05) * 0.05),
        min(1.0, math.ceil((float(raw_values.max()) + 0.05) / 0.05) * 0.05),
    )
    robust_range = (
        0.0,
        min(
            1.0,
            max(0.05, math.ceil(float(robust_values.max()) / 0.05) * 0.05),
        ),
    )
    for model_name in models:
        plot_model_win_rate_heatmap(
            summary_frame,
            model_name,
            output_dir,
            raw_range,
            robust_range,
            args.dpi,
        )
    plot_strict_ecdf(frames, thresholds, models, output_dir, zoom, args.dpi)
    plot_review_scope(summary_frame, models, output_dir, args.dpi)

    edges = np.linspace(zoom[0], zoom[1], zoom_bins + 1)
    histogram = histogram_rows(frames, models, edges)
    write_csv_atomic(
        output_dir / "delta_histogram_bins.csv",
        histogram,
        ("model", "scope", "bin_left", "bin_right", "count"),
    )
    metadata = {
        "schema_version": "1.0",
        "dataset": "PinPoint V7",
        "input_path": str(input_path),
        "diagnostics_root": str(diagnostics_root),
        "models": models,
        "negative_count_per_model": {
            model_name: len(frames[model_name]) for model_name in models
        },
        "construction_mode_note": (
            "V7 has one protocol: text-only generation followed by vision-grounded "
            "selection. Analysis dimensions are operation, 3x3 cell, and review scope."
        ),
        "delta_variable": (
            "per-negative mean over original Positive targets of "
            "sim(f(R,m+),t)-sim(f(R,m-),t)"
        ),
        "theoretical_range": [-2.0, 2.0],
        "empirical_zoom": list(zoom),
        "thresholds": {
            model_name: {
                "value": thresholds[model_name],
                "bootstrap_ci": model_metadata[model_name][
                    "threshold_bootstrap_ci"
                ],
            }
            for model_name in models
        },
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "cell_counts": dict(
            sorted(Counter(frames[models[0]]["cell"].tolist()).items())
        ),
    }
    write_json_atomic(output_dir / "metadata.json", metadata)
    (output_dir / "summary.md").write_text(
        make_summary(summary_frame, thresholds, models, zoom),
        encoding="utf-8",
    )
    print(f"[done] wrote PinPoint V7 Delta analysis to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
