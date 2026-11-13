#!/usr/bin/env python3
"""Calibrate Delta uncertainty from positive-query paraphrases.

For each model, this script encodes only the accepted positive paraphrases.
Existing target embeddings and negative scores are reused from a completed
``run_diagnostics.py`` output.

The calibration variable is

    epsilon = sim(f(R, paraphrase(m+)), t) - sim(f(R, m+), t).

Its pair-balanced empirical interval [L, U] turns each original Delta into the
wording-variation interval [Delta + L, Delta + U].  A negative is a stable win
only when the lower endpoint is positive, a stable loss only when the upper
endpoint is negative, and otherwise is marked uncertain.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from common import read_jsonl, sha256_file, write_json_atomic
from extract_embeddings import (
    DEFAULT_BATCH_SIZES,
    SUPPORTED_MODELS,
    QueryItem,
    Sample,
    cache_path,
    extract_queries,
    load_cached_vector,
    load_dataset,
    make_adapter,
)


CALIBRATION_VERSION = "2"


@dataclass(frozen=True)
class ParaphraseItem:
    sample_id: str
    pair_id: str
    paraphrase_id: str
    reference_path: str
    text: str
    rewrite_strategy: str
    length_ratio: float | None
    token_jaccard: float | None

    @property
    def logical_cache_key(self) -> str:
        return (
            f"positive-paraphrase:v{CALIBRATION_VERSION}:{self.sample_id}:"
            f"{self.paraphrase_id}:{self.reference_path}:{self.text}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate positive-paraphrase robustness for one diagnostic model."
    )
    parser.add_argument("--input", required=True, type=Path, help="CIR-QDiag final.jsonl")
    parser.add_argument("--paraphrases", required=True, type=Path)
    parser.add_argument("--diagnostics-dir", required=True, type=Path)
    parser.add_argument("--model-name", required=True, choices=tuple(SUPPORTED_MODELS))
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--interval-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_paraphrase_items(
    path: Path,
    samples: Sequence[Sample],
) -> tuple[list[ParaphraseItem], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Paraphrase JSONL does not exist: {path}")

    sample_by_id = {sample.sample_id: sample for sample in samples}
    records: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(path):
        sample_id = str(record["sample_id"])
        if sample_id in records:
            raise ValueError(f"Duplicate paraphrase sample_id: {sample_id}")
        records[sample_id] = record

    missing_records = sorted(set(sample_by_id) - set(records))
    if missing_records:
        raise ValueError(
            "Paraphrase file is missing final samples: " + ", ".join(missing_records)
        )

    ignored_samples = sorted(set(records) - set(sample_by_id))
    items: list[ParaphraseItem] = []
    seen_ids: set[str] = set()
    accepted_counts: dict[str, int] = {}
    for sample in samples:
        record = records[sample.sample_id]
        if str(record["pair_id"]) != sample.pair_id:
            raise ValueError(f"pair_id mismatch for {sample.sample_id}")
        if str(record["positive_text"]) != sample.positive_text:
            raise ValueError(f"positive_text mismatch for {sample.sample_id}")
        if str(record["reference"]["image_path"]) != sample.reference_path:
            raise ValueError(f"reference path mismatch for {sample.sample_id}")
        if str(record["target"]["image_path"]) != sample.target_path:
            raise ValueError(f"target path mismatch for {sample.sample_id}")

        accepted = list(record.get("paraphrases") or [])
        accepted_counts[sample.sample_id] = len(accepted)
        for raw in accepted:
            paraphrase_id = str(raw["paraphrase_id"])
            if paraphrase_id in seen_ids:
                raise ValueError(f"Duplicate paraphrase_id: {paraphrase_id}")
            seen_ids.add(paraphrase_id)
            text = str(raw["text"]).strip()
            if not text:
                raise ValueError(f"Empty paraphrase text: {paraphrase_id}")
            if text.casefold() == sample.positive_text.strip().casefold():
                raise ValueError(f"Paraphrase is identical to source: {paraphrase_id}")
            metrics = raw.get("metrics") or {}
            items.append(
                ParaphraseItem(
                    sample_id=sample.sample_id,
                    pair_id=sample.pair_id,
                    paraphrase_id=paraphrase_id,
                    reference_path=sample.reference_path,
                    text=text,
                    rewrite_strategy=str(raw.get("rewrite_strategy") or ""),
                    length_ratio=_optional_float(metrics.get("length_ratio")),
                    token_jaccard=_optional_float(metrics.get("token_jaccard")),
                )
            )

    count_histogram: dict[str, int] = {}
    for count in accepted_counts.values():
        key = str(count)
        count_histogram[key] = count_histogram.get(key, 0) + 1
    coverage = {
        "final_sample_count": len(samples),
        "covered_sample_count": sum(count > 0 for count in accepted_counts.values()),
        "accepted_paraphrase_count": len(items),
        "accepted_count_histogram": count_histogram,
        "samples_without_accepted_paraphrases": sorted(
            sample_id for sample_id, count in accepted_counts.items() if count == 0
        ),
        "ignored_nonfinal_samples": ignored_samples,
    }
    return items, coverage


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"Expected a finite numeric value, got {value!r}")
    return result


def runtime_query_cache_key(
    logical_key: str,
    *,
    model_name: str,
    model_path: Path,
    device: str,
    dtype: str,
) -> str:
    """Keep calibration caches isolated across runtime/model configurations."""

    return (
        f"{logical_key}:model={model_name}:model_path={model_path.resolve()}:"
        f"device={device}:dtype={dtype}"
    )


def original_runtime_logical_key(sample: Sample) -> str:
    return (
        f"positive-calibration:v{CALIBRATION_VERSION}:{sample.sample_id}:"
        f"{sample.reference_path}:{sample.positive_text}"
    )


def weighted_quantile(
    values: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray,
    quantile: float,
) -> float:
    """Return an inverse-CDF weighted empirical quantile."""

    x = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if x.ndim != 1 or w.ndim != 1 or x.size != w.size or x.size == 0:
        raise ValueError("values and weights must be non-empty one-dimensional arrays")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    if not np.isfinite(x).all() or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("values/weights must be finite and weights non-negative")
    total = float(w.sum())
    if total <= 0:
        raise ValueError("weights must have positive total mass")
    order = np.argsort(x, kind="stable")
    sorted_values = x[order]
    cumulative = np.cumsum(w[order])
    target = quantile * total
    index = int(np.searchsorted(cumulative, target, side="left"))
    return float(sorted_values[min(index, sorted_values.size - 1)])


def pair_equal_weights(groups: Sequence[str] | np.ndarray) -> np.ndarray:
    group_array = np.asarray(groups, dtype=object)
    if group_array.ndim != 1 or group_array.size == 0:
        raise ValueError("groups must be a non-empty one-dimensional array")
    unique, inverse, counts = np.unique(
        group_array.astype(str), return_inverse=True, return_counts=True
    )
    del unique
    return 1.0 / counts[inverse].astype(np.float64)


def pair_balanced_interval(
    values: Sequence[float] | np.ndarray,
    groups: Sequence[str] | np.ndarray,
    level: float,
) -> tuple[float, float]:
    if not 0.0 < level < 1.0:
        raise ValueError("interval level must lie in (0, 1)")
    alpha = (1.0 - level) / 2.0
    weights = pair_equal_weights(groups)
    return (
        weighted_quantile(values, weights, alpha),
        weighted_quantile(values, weights, 1.0 - alpha),
    )


def bootstrap_calibration(
    shifts: np.ndarray,
    groups: np.ndarray,
    *,
    interval_level: float,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    groups = np.asarray(groups, dtype=object).astype(str)
    shifts = np.asarray(shifts, dtype=np.float64)
    unique_groups = np.unique(groups)
    grouped_values = [shifts[groups == group] for group in unique_groups]
    point_low, point_high = pair_balanced_interval(
        shifts, groups, interval_level
    )
    symmetric_tau = weighted_quantile(
        np.abs(shifts), pair_equal_weights(groups), interval_level
    )

    rng = np.random.default_rng(seed)
    lows = np.empty(replicates, dtype=np.float64)
    highs = np.empty(replicates, dtype=np.float64)
    taus = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        selected = rng.integers(0, len(grouped_values), size=len(grouped_values))
        sampled_values = np.concatenate([grouped_values[index] for index in selected])
        sampled_groups = np.concatenate(
            [
                np.full(grouped_values[index].size, draw_index, dtype=np.int64)
                for draw_index, index in enumerate(selected)
            ]
        )
        lows[replicate], highs[replicate] = pair_balanced_interval(
            sampled_values, sampled_groups, interval_level
        )
        taus[replicate] = weighted_quantile(
            np.abs(sampled_values),
            pair_equal_weights(sampled_groups),
            interval_level,
        )

    endpoint_confidence = 0.95
    endpoint_alpha = (1.0 - endpoint_confidence) / 2.0
    return {
        "interval_level": interval_level,
        "shift_interval_low": point_low,
        "shift_interval_high": point_high,
        "shift_interval_low_bootstrap_ci": [
            float(np.quantile(lows, endpoint_alpha)),
            float(np.quantile(lows, 1.0 - endpoint_alpha)),
        ],
        "shift_interval_high_bootstrap_ci": [
            float(np.quantile(highs, endpoint_alpha)),
            float(np.quantile(highs, 1.0 - endpoint_alpha)),
        ],
        "symmetric_abs_shift_threshold": symmetric_tau,
        "symmetric_threshold_bootstrap_ci": [
            float(np.quantile(taus, endpoint_alpha)),
            float(np.quantile(taus, 1.0 - endpoint_alpha)),
        ],
        "bootstrap_replicates": replicates,
        "bootstrap_confidence_level": endpoint_confidence,
        "calibration_pair_count": int(unique_groups.size),
        "calibration_paraphrase_count": int(shifts.size),
        "pair_weighting": "each pair has total weight 1",
    }


def leave_one_pair_out_intervals(
    shifts: np.ndarray,
    shift_groups: np.ndarray,
    evaluation_groups: Iterable[str],
    level: float,
) -> dict[str, tuple[float, float]]:
    shifts = np.asarray(shifts, dtype=np.float64)
    shift_groups = np.asarray(shift_groups, dtype=object).astype(str)
    result: dict[str, tuple[float, float]] = {}
    for group in sorted(set(str(value) for value in evaluation_groups)):
        mask = shift_groups != group
        if np.unique(shift_groups[mask]).size < 2:
            raise ValueError("Too few calibration pairs after leave-one-pair-out")
        result[group] = pair_balanced_interval(
            shifts[mask], shift_groups[mask], level
        )
    return result


def classify_intervals(
    deltas: Sequence[float] | np.ndarray,
    interval_lows: Sequence[float] | np.ndarray,
    interval_highs: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    delta = np.asarray(deltas, dtype=np.float64)
    lower = delta + np.asarray(interval_lows, dtype=np.float64)
    upper = delta + np.asarray(interval_highs, dtype=np.float64)
    if delta.shape != lower.shape or delta.shape != upper.shape:
        raise ValueError("delta and interval endpoint arrays must have matching shapes")
    state = np.full(delta.shape, "uncertain", dtype=object)
    state[lower > 0.0] = "stable_win"
    state[upper < 0.0] = "stable_loss"
    return lower, upper, state


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _load_diagnostic_state(
    diagnostics_dir: Path,
    model_name: str,
    input_path: Path,
    samples: Sequence[Sample],
) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, Any]]:
    metadata_path = diagnostics_dir / "metadata.json"
    features_path = diagnostics_dir / "features.npz"
    scores_path = diagnostics_dir / "scores.csv"
    for path in (metadata_path, features_path, scores_path):
        if not path.is_file():
            raise FileNotFoundError(f"Completed diagnostic artifact is missing: {path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not metadata.get("completed"):
        raise RuntimeError(f"Diagnostics are incomplete: {diagnostics_dir}")
    if metadata.get("model_name") != model_name:
        raise RuntimeError(
            f"Diagnostic model mismatch: {metadata.get('model_name')} != {model_name}"
        )
    if metadata.get("input_sha256") != sha256_file(input_path):
        raise RuntimeError("Diagnostic input SHA256 differs from the requested final.jsonl")

    with np.load(features_path, allow_pickle=False) as loaded:
        features = {key: np.asarray(loaded[key]) for key in loaded.files}
    expected_samples = len(samples)
    for key in ("q_positive", "target"):
        if features[key].shape[0] != expected_samples:
            raise RuntimeError(
                f"{key} has {features[key].shape[0]} samples, expected {expected_samples}"
            )
    scores = pd.read_csv(scores_path)
    scores["sample_id"] = scores["sample_id"].astype(str)
    scores["pair_id"] = scores["pair_id"].astype(str)
    return features, scores, metadata


def build_paraphrase_score_frame(
    items: Sequence[ParaphraseItem],
    samples: Sequence[Sample],
    cache_root: Path,
    features: dict[str, np.ndarray],
    scores: pd.DataFrame,
    *,
    model_name: str,
    model_path: Path,
    device: str,
    dtype: str,
) -> pd.DataFrame:
    sample_index = {sample.sample_id: sample.sample_index for sample in samples}
    sample_by_id = {sample.sample_id: sample for sample in samples}
    positive_by_sample = scores.groupby("sample_id", sort=False)["positive_score"]
    stored_positive_score = positive_by_sample.first().to_dict()
    positive_spread = positive_by_sample.max() - positive_by_sample.min()
    if float(positive_spread.max()) > 1e-5:
        raise AssertionError("Original positive score is not constant within a sample")

    runtime_original_score: dict[str, float] = {}
    for sample in samples:
        logical_key = original_runtime_logical_key(sample)
        key = runtime_query_cache_key(
            logical_key,
            model_name=model_name,
            model_path=model_path,
            device=device,
            dtype=dtype,
        )
        vector = load_cached_vector(cache_path(cache_root, "query", key))
        target = np.asarray(features["target"][sample.sample_index], dtype=np.float32)
        runtime_original_score[sample.sample_id] = float(np.dot(vector, target))

    rows: list[dict[str, Any]] = []
    for item in items:
        key = runtime_query_cache_key(
            item.logical_cache_key,
            model_name=model_name,
            model_path=model_path,
            device=device,
            dtype=dtype,
        )
        vector = load_cached_vector(
            cache_path(cache_root, "query", key)
        )
        target = np.asarray(
            features["target"][sample_index[item.sample_id]], dtype=np.float32
        )
        if vector.shape != target.shape:
            raise ValueError(
                f"Embedding dimension mismatch for {item.paraphrase_id}: "
                f"{vector.shape} != {target.shape}"
            )
        paraphrase_score = float(np.dot(vector, target))
        original_score = float(stored_positive_score[item.sample_id])
        reencoded_score = float(runtime_original_score[item.sample_id])
        rows.append(
            {
                "sample_id": item.sample_id,
                "pair_id": item.pair_id,
                "paraphrase_id": item.paraphrase_id,
                "paraphrase_text": item.text,
                "rewrite_strategy": item.rewrite_strategy,
                "length_ratio": item.length_ratio,
                "token_jaccard": item.token_jaccard,
                "stored_original_positive_score": original_score,
                "reencoded_original_positive_score": reencoded_score,
                "original_rescore_drift": reencoded_score - original_score,
                "paraphrase_positive_score": paraphrase_score,
                "positive_score_shift": paraphrase_score - reencoded_score,
            }
        )
    return pd.DataFrame(rows)


def build_robust_negative_frame(
    scores: pd.DataFrame,
    paraphrase_scores: pd.DataFrame,
    *,
    interval_level: float,
) -> pd.DataFrame:
    result = scores.copy()
    shifts = paraphrase_scores["positive_score_shift"].to_numpy(dtype=np.float64)
    shift_groups = paraphrase_scores["pair_id"].astype(str).to_numpy()
    intervals = leave_one_pair_out_intervals(
        shifts,
        shift_groups,
        result["pair_id"].astype(str),
        interval_level,
    )
    result["paraphrase_shift_interval_low"] = [
        intervals[str(pair_id)][0] for pair_id in result["pair_id"]
    ]
    result["paraphrase_shift_interval_high"] = [
        intervals[str(pair_id)][1] for pair_id in result["pair_id"]
    ]
    lower, upper, state = classify_intervals(
        result["delta"],
        result["paraphrase_shift_interval_low"],
        result["paraphrase_shift_interval_high"],
    )
    result["calibrated_delta_low"] = lower
    result["calibrated_delta_high"] = upper
    result["calibrated_state"] = state
    result["original_win"] = result["delta"] > 0.0

    shifts_by_sample = {
        sample_id: group["positive_score_shift"].to_numpy(dtype=np.float64)
        for sample_id, group in paraphrase_scores.groupby("sample_id", sort=False)
    }
    observed_count: list[int] = []
    observed_min: list[float] = []
    observed_mean: list[float] = []
    observed_max: list[float] = []
    observed_win_fraction: list[float] = []
    observed_state: list[str] = []
    for row in result.itertuples(index=False):
        sample_shifts = shifts_by_sample.get(str(row.sample_id))
        if sample_shifts is None:
            observed_count.append(0)
            observed_min.append(np.nan)
            observed_mean.append(np.nan)
            observed_max.append(np.nan)
            observed_win_fraction.append(np.nan)
            observed_state.append("unavailable")
            continue
        variant_deltas = np.concatenate(
            (np.asarray([float(row.delta)]), float(row.delta) + sample_shifts)
        )
        observed_count.append(int(sample_shifts.size))
        observed_min.append(float(variant_deltas.min()))
        observed_mean.append(float(variant_deltas.mean()))
        observed_max.append(float(variant_deltas.max()))
        observed_win_fraction.append(float(np.mean(variant_deltas > 0.0)))
        if float(variant_deltas.min()) > 0.0:
            observed_state.append("stable_win")
        elif float(variant_deltas.max()) < 0.0:
            observed_state.append("stable_loss")
        else:
            observed_state.append("uncertain")

    result["observed_paraphrase_count"] = observed_count
    result["observed_delta_min"] = observed_min
    result["observed_delta_mean"] = observed_mean
    result["observed_delta_max"] = observed_max
    result["observed_win_fraction"] = observed_win_fraction
    result["observed_state"] = observed_state
    return result


def _pair_balanced_mean(frame: pd.DataFrame, column: str) -> float:
    pair_values = frame.groupby("pair_id", sort=False)[column].mean()
    return float(pair_values.mean())


def _pair_bootstrap_interval(
    frame: pd.DataFrame,
    column: str,
    *,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    pair_values = (
        frame.groupby("pair_id", sort=False)[column].mean().to_numpy(dtype=np.float64)
    )
    if pair_values.size == 0:
        return (float("nan"), float("nan"))
    indices = rng.integers(
        0, pair_values.size, size=(replicates, pair_values.size)
    )
    estimates = pair_values[indices].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def aggregate_robustness(
    frame: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    seed: int,
) -> pd.DataFrame:
    strict = frame["edit_operation"] != "delete"
    subset_specs: list[tuple[str, np.ndarray]] = [
        ("all", np.ones(len(frame), dtype=bool)),
        ("strict", strict.to_numpy()),
        ("add", (frame["edit_operation"] == "add").to_numpy()),
        ("substitute", (frame["edit_operation"] == "substitute").to_numpy()),
        ("delete_descriptive", (frame["edit_operation"] == "delete").to_numpy()),
        (
            "text_strict",
            (
                strict
                & (frame["construction_mode"] == "text_only_editable")
            ).to_numpy(),
        ),
        (
            "vision_strict",
            (
                strict
                & (frame["construction_mode"] == "vision_grounded")
            ).to_numpy(),
        ),
    ]
    for semantic in ("object", "attribute", "predicate"):
        for operation in ("add", "substitute"):
            subset_specs.append(
                (
                    f"{semantic}|{operation}",
                    (
                        (frame["semantic_target"] == semantic)
                        & (frame["edit_operation"] == operation)
                    ).to_numpy(),
                )
            )
            for mode in ("text_only_editable", "vision_grounded"):
                subset_specs.append(
                    (
                        f"{mode}|{semantic}|{operation}",
                        (
                            (frame["construction_mode"] == mode)
                            & (frame["semantic_target"] == semantic)
                            & (frame["edit_operation"] == operation)
                        ).to_numpy(),
                    )
                )

    working = frame.copy()
    working["calibrated_stable_win"] = (
        working["calibrated_state"] == "stable_win"
    ).astype(float)
    working["calibrated_uncertain"] = (
        working["calibrated_state"] == "uncertain"
    ).astype(float)
    working["calibrated_stable_loss"] = (
        working["calibrated_state"] == "stable_loss"
    ).astype(float)
    working["observed_available"] = (
        working["observed_state"] != "unavailable"
    ).astype(float)
    working["observed_stable_win"] = np.where(
        working["observed_available"] > 0,
        (working["observed_state"] == "stable_win").astype(float),
        np.nan,
    )
    working["observed_uncertain"] = np.where(
        working["observed_available"] > 0,
        (working["observed_state"] == "uncertain").astype(float),
        np.nan,
    )
    working["observed_stable_loss"] = np.where(
        working["observed_available"] > 0,
        (working["observed_state"] == "stable_loss").astype(float),
        np.nan,
    )

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for subset_name, mask in subset_specs:
        subset = working.loc[mask].copy()
        if subset.empty:
            continue
        delta_ci = _pair_bootstrap_interval(
            subset, "delta", replicates=bootstrap_replicates, rng=rng
        )
        original_win_ci = _pair_bootstrap_interval(
            subset, "original_win", replicates=bootstrap_replicates, rng=rng
        )
        stable_win_ci = _pair_bootstrap_interval(
            subset,
            "calibrated_stable_win",
            replicates=bootstrap_replicates,
            rng=rng,
        )
        observed_subset = subset.dropna(subset=["observed_win_fraction"])
        observed_variant_win_ci = _pair_bootstrap_interval(
            observed_subset,
            "observed_win_fraction",
            replicates=bootstrap_replicates,
            rng=rng,
        )
        rows.append(
            {
                "subset": subset_name,
                "negative_count": len(subset),
                "pair_count": subset["pair_id"].nunique(),
                "pair_balanced_mean_delta": _pair_balanced_mean(subset, "delta"),
                "mean_delta_ci_low": delta_ci[0],
                "mean_delta_ci_high": delta_ci[1],
                "pair_balanced_original_win_rate": _pair_balanced_mean(
                    subset, "original_win"
                ),
                "original_win_ci_low": original_win_ci[0],
                "original_win_ci_high": original_win_ci[1],
                "pair_balanced_calibrated_stable_win_rate": _pair_balanced_mean(
                    subset, "calibrated_stable_win"
                ),
                "calibrated_stable_win_ci_low": stable_win_ci[0],
                "calibrated_stable_win_ci_high": stable_win_ci[1],
                "pair_balanced_calibrated_uncertain_rate": _pair_balanced_mean(
                    subset, "calibrated_uncertain"
                ),
                "pair_balanced_calibrated_stable_loss_rate": _pair_balanced_mean(
                    subset, "calibrated_stable_loss"
                ),
                "pair_balanced_mean_calibrated_delta_low": _pair_balanced_mean(
                    subset, "calibrated_delta_low"
                ),
                "pair_balanced_mean_calibrated_delta_high": _pair_balanced_mean(
                    subset, "calibrated_delta_high"
                ),
                "observed_pair_coverage_rate": float(
                    subset.loc[
                        subset["observed_available"] > 0, "pair_id"
                    ].nunique()
                    / subset["pair_id"].nunique()
                ),
                "pair_balanced_observed_variant_win_rate": _pair_balanced_mean(
                    observed_subset,
                    "observed_win_fraction",
                ),
                "observed_variant_win_ci_low": observed_variant_win_ci[0],
                "observed_variant_win_ci_high": observed_variant_win_ci[1],
                "pair_balanced_observed_mean_delta": _pair_balanced_mean(
                    observed_subset,
                    "observed_delta_mean",
                ),
                "pair_balanced_observed_worst_delta": _pair_balanced_mean(
                    observed_subset,
                    "observed_delta_min",
                ),
                "pair_balanced_observed_stable_win_rate": _pair_balanced_mean(
                    subset.dropna(subset=["observed_stable_win"]),
                    "observed_stable_win",
                ),
                "pair_balanced_observed_uncertain_rate": _pair_balanced_mean(
                    subset.dropna(subset=["observed_uncertain"]),
                    "observed_uncertain",
                ),
                "pair_balanced_observed_stable_loss_rate": _pair_balanced_mean(
                    subset.dropna(subset=["observed_stable_loss"]),
                    "observed_stable_loss",
                ),
            }
        )
    return pd.DataFrame(rows)


def _cleanup_adapter(adapter: Any) -> None:
    del adapter
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def main() -> None:
    args = parse_args()
    if not 0.0 < args.interval_level < 1.0:
        raise ValueError("--interval-level must lie in (0, 1)")
    if args.bootstrap_replicates <= 0:
        raise ValueError("--bootstrap-replicates must be positive")
    batch_size = args.batch_size or DEFAULT_BATCH_SIZES[args.model_name]
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    input_path = args.input.resolve()
    paraphrase_path = args.paraphrases.resolve()
    diagnostics_dir = args.diagnostics_dir.resolve()
    model_path = (args.model_path or SUPPORTED_MODELS[args.model_name]).resolve()
    samples, _ = load_dataset(input_path, None)
    items, coverage = load_paraphrase_items(paraphrase_path, samples)
    if not items:
        raise ValueError("No accepted paraphrases overlap the final diagnostic samples")
    features, scores, diagnostic_metadata = _load_diagnostic_state(
        diagnostics_dir, args.model_name, input_path, samples
    )

    cache_root = diagnostics_dir / "cache"
    paraphrase_query_items = [
        QueryItem(
            cache_key=runtime_query_cache_key(
                item.logical_cache_key,
                model_name=args.model_name,
                model_path=model_path,
                device=args.device,
                dtype=args.dtype,
            ),
            query_id=f"pp:{item.paraphrase_id}",
            image_path=item.reference_path,
            text=item.text,
        )
        for item in items
    ]
    original_query_items = [
        QueryItem(
            cache_key=runtime_query_cache_key(
                original_runtime_logical_key(sample),
                model_name=args.model_name,
                model_path=model_path,
                device=args.device,
                dtype=args.dtype,
            ),
            query_id=f"po:{sample.sample_id}",
            image_path=sample.reference_path,
            text=sample.positive_text,
        )
        for sample in samples
    ]
    query_items = original_query_items + paraphrase_query_items
    pending = [
        item
        for item in query_items
        if args.force
        or not cache_path(cache_root, "query", item.cache_key).is_file()
    ]
    print(
        f"[info] model={args.model_name} final_samples={len(samples)} "
        f"covered_samples={coverage['covered_sample_count']} "
        f"paraphrases={len(items)} calibration_queries={len(query_items)} "
        f"cached={len(query_items) - len(pending)} "
        f"pending={len(pending)}"
    )
    if pending:
        adapter_args = argparse.Namespace(
            output_dir=diagnostics_dir,
            model_name=args.model_name,
            device=args.device,
            dtype=args.dtype,
        )
        adapter = make_adapter(adapter_args, model_path)
        extract_queries(
            adapter,
            query_items,
            cache_root,
            batch_size,
            resume=True,
            force=args.force,
        )
        _cleanup_adapter(adapter)

    paraphrase_scores = build_paraphrase_score_frame(
        items,
        samples,
        cache_root,
        features,
        scores,
        model_name=args.model_name,
        model_path=model_path,
        device=args.device,
        dtype=args.dtype,
    )
    shifts = paraphrase_scores["positive_score_shift"].to_numpy(dtype=np.float64)
    shift_groups = paraphrase_scores["pair_id"].astype(str).to_numpy()
    calibration = bootstrap_calibration(
        shifts,
        shift_groups,
        interval_level=args.interval_level,
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    pair_means = paraphrase_scores.groupby("pair_id")[
        "positive_score_shift"
    ].mean()
    drift_by_sample = paraphrase_scores.groupby("sample_id")[
        "original_rescore_drift"
    ].first()
    calibration.update(
        {
            "schema_version": "1.0",
            "calibration_version": CALIBRATION_VERSION,
            "model_name": args.model_name,
            "input_path": str(input_path),
            "input_sha256": sha256_file(input_path),
            "paraphrase_path": str(paraphrase_path),
            "paraphrase_sha256": sha256_file(paraphrase_path),
            "diagnostics_dir": str(diagnostics_dir),
            "diagnostic_input_sha256": diagnostic_metadata["input_sha256"],
            "model_path": str(model_path),
            "device": args.device,
            "dtype": args.dtype,
            "batch_size": batch_size,
            "seed": args.seed,
            "python": platform.python_version(),
            "pair_balanced_mean_shift": float(pair_means.mean()),
            "pair_balanced_median_shift": weighted_quantile(
                shifts, pair_equal_weights(shift_groups), 0.5
            ),
            "original_rescore_drift": {
                "mean": float(drift_by_sample.mean()),
                "mean_absolute": float(drift_by_sample.abs().mean()),
                "max_absolute": float(drift_by_sample.abs().max()),
                "purpose": (
                    "diagnostic only; paraphrase shifts are measured against the "
                    "same-run re-encoded original query"
                ),
            },
            "coverage": coverage,
            "classification": {
                "method": "leave-one-pair-out calibration interval",
                "stable_win": "original_delta + shift_interval_low > 0",
                "stable_loss": "original_delta + shift_interval_high < 0",
                "otherwise": "uncertain",
            },
        }
    )

    robust_scores = build_robust_negative_frame(
        scores,
        paraphrase_scores,
        interval_level=args.interval_level,
    )
    summary = aggregate_robustness(
        robust_scores,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    output_dir = diagnostics_dir / "paraphrase_robustness"
    _atomic_csv(paraphrase_scores, output_dir / "paraphrase_scores.csv")
    _atomic_csv(robust_scores, output_dir / "robust_negative_scores.csv")
    _atomic_csv(summary, output_dir / "summary.csv")
    write_json_atomic(output_dir / "calibration.json", calibration)
    print(
        f"[done] interval=[{calibration['shift_interval_low']:+.6f}, "
        f"{calibration['shift_interval_high']:+.6f}] "
        f"outputs={output_dir}"
    )


if __name__ == "__main__":
    main()
