#!/usr/bin/env python3
"""Evaluate PinPoint Delta wins, paraphrase calibration, and random separation.

This is a score-only diagnostic. It encodes the PinPoint V7 positive-image
subset and all positive/generated-negative queries, but it does not encode or
search the full PinPoint corpus.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import re
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from common import write_json_atomic
from pinpoint_full_diagnostics import (
    GroupSpec,
    QuerySpec,
    bootstrap_group_balanced_quantile,
    group_balanced_quantile,
    load_or_build_query_embeddings,
    load_v7_dataset,
)
from trained_clip_adapter import OriginalCLIPAdapter, TrainedCLIPAdapter


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]
CIR_PLAY_ROOT = WORKSPACE_ROOT / "CIR_PLAY"
DEFAULT_INPUT = (
    SCRIPT_DIR.parent
    / "cir_qdiag_openai"
    / "outputs"
    / "pinpoint_pilot_500_v7"
    / "final.jsonl"
)
DEFAULT_PINPOINT_ROOT = WORKSPACE_ROOT / "pinpoint-dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="NAME=PATH",
    )
    parser.add_argument(
        "--pretrained-clip",
        action="append",
        default=[],
        metavar="NAME=BACKBONE_DIR",
        help="Evaluate ordinary pretrained CLIP-L sum composition without a checkpoint.",
    )
    parser.add_argument(
        "--pretrained-model",
        action="append",
        default=[],
        metavar="NAME=MODEL_DIR",
        help=(
            "Evaluate an original model supported by CIR_PLAY adapters, such as "
            "BGE-VL-MLLM-S2; repeat for multiple models."
        ),
    )
    parser.add_argument(
        "--adapter",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Evaluate a trained checkpoint through the CIR_PLAY adapter registry.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pinpoint-root", type=Path, default=DEFAULT_PINPOINT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--prefetch-batches", type=int, default=4)
    parser.add_argument("--processor-workers", type=int, default=2)
    parser.add_argument("--processor-prefetch-batches", type=int, default=2)
    parser.add_argument("--flush-interval", type=int, default=16)
    parser.add_argument("--random-samples", type=int, default=10)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_checkpoints(values: Sequence[str]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    names: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid --checkpoint {value!r}; expected NAME=PATH")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError(f"unsafe checkpoint name: {name!r}")
        if name in names:
            raise ValueError(f"duplicate checkpoint name: {name}")
        path = Path(raw_path.strip()).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        names.add(name)
        result.append((name, path))
    return result


def parse_pretrained_clips(values: Sequence[str]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    names: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid --pretrained-clip {value!r}; expected NAME=PATH")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError(f"unsafe pretrained CLIP name: {name!r}")
        if name in names:
            raise ValueError(f"duplicate pretrained CLIP name: {name}")
        path = Path(raw_path.strip()).resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        names.add(name)
        result.append((name, path))
    return result


def parse_pretrained_models(values: Sequence[str]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    names: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid --pretrained-model {value!r}; expected NAME=MODEL_DIR")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError(f"unsafe pretrained model name: {name!r}")
        if name in names:
            raise ValueError(f"duplicate pretrained model name: {name}")
        path = Path(raw_path.strip()).resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        names.add(name)
        result.append((name, path))
    return result


def parse_adapters(values: Sequence[str]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    names: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid --adapter {value!r}; expected NAME=PATH")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError(f"unsafe adapter name: {name!r}")
        if name in names:
            raise ValueError(f"duplicate adapter name: {name}")
        path = Path(raw_path.strip()).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        names.add(name)
        result.append((name, path))
    return result


def _rng(seed: int, key: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}\x1f{key}".encode()).digest()
    derived = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return np.random.default_rng(derived)


def _sample_indices(
    candidates: Sequence[int],
    *,
    count: int,
    seed: int,
    key: str,
) -> np.ndarray:
    values = np.asarray(sorted(set(int(value) for value in candidates)), dtype=np.int64)
    if values.size < count:
        raise ValueError(f"{key} has only {values.size} unrelated candidates, need {count}")
    return _rng(seed, key).choice(values, size=count, replace=False)


def _hierarchical_mean(frame: pd.DataFrame, column: str) -> float:
    by_instruction = frame.groupby(["group_id", "instruction_id"], sort=False)[
        column
    ].mean()
    by_group = by_instruction.groupby(level="group_id", sort=False).mean()
    return float(by_group.mean())


def _bootstrap_hierarchical_mean(
    frame: pd.DataFrame,
    column: str,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    by_instruction = frame.groupby(["group_id", "instruction_id"], sort=False)[
        column
    ].mean()
    values = (
        by_instruction.groupby(level="group_id", sort=False)
        .mean()
        .to_numpy(dtype=np.float64)
    )
    rng = _rng(seed, f"bootstrap:{column}")
    indices = rng.integers(0, values.size, size=(replicates, values.size))
    estimates = values[indices].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def calibrate_threshold(
    groups: Sequence[GroupSpec],
    queries: Sequence[QuerySpec],
    query_embeddings: np.ndarray,
    image_embeddings: np.ndarray,
    image_index: dict[str, int],
    *,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    positive_query_index = {
        (query.group_id, query.instruction_id): query.query_index
        for query in queries
        if query.query_kind == "positive"
    }
    perturbations: dict[str, np.ndarray] = {}
    for group in groups:
        query_rows = [
            positive_query_index[(group.group_id, instruction_id)]
            for instruction_id in group.instruction_ids
        ]
        image_rows = [image_index[value] for value in group.positive_ids]
        scores = (
            np.asarray(query_embeddings[query_rows], dtype=np.float32)
            @ np.asarray(image_embeddings[image_rows], dtype=np.float32).T
        )
        differences = [
            np.abs(scores[first] - scores[second])
            for first, second in combinations(range(len(query_rows)), 2)
        ]
        perturbations[group.group_id] = (
            np.concatenate(differences)
            if differences
            else np.empty(0, dtype=np.float64)
        )
    threshold = group_balanced_quantile(perturbations, 0.95)
    threshold_ci = bootstrap_group_balanced_quantile(
        perturbations,
        quantile=0.95,
        replicates=bootstrap_replicates,
        seed=seed,
    )
    return {
        "method": (
            "group-balanced 95th percentile of absolute native positive-"
            "paraphrase target-score differences"
        ),
        "formula": "|sim(f(R,m_i+),t)-sim(f(R,m_j+),t)|",
        "normal_perturbation_threshold": threshold,
        "threshold_bootstrap_ci": list(threshold_ci),
        "calibration_group_count": len(perturbations),
        "calibration_observation_count": int(
            sum(len(values) for values in perturbations.values())
        ),
        "group_weighting": "each PinPoint rewrite group has total weight 1",
    }


def score_negatives(
    groups: Sequence[GroupSpec],
    queries: Sequence[QuerySpec],
    query_embeddings: np.ndarray,
    image_embeddings: np.ndarray,
    image_ids: Sequence[str],
    *,
    threshold: float,
    random_samples: int,
    seed: int,
) -> pd.DataFrame:
    group_map = {group.group_id: group for group in groups}
    image_index = {str(value): index for index, value in enumerate(image_ids)}
    positive_query_index = {
        (query.group_id, query.instruction_id): query.query_index
        for query in queries
        if query.query_kind == "positive"
    }
    positive_query_rows = [
        query.query_index for query in queries if query.query_kind == "positive"
    ]
    negative_query_rows = [
        query.query_index for query in queries if query.query_kind == "negative"
    ]
    query_by_index = {query.query_index: query for query in queries}
    all_image_rows = list(range(len(image_ids)))
    rows: list[dict[str, Any]] = []
    for query in queries:
        if query.query_kind != "negative":
            continue
        group = group_map[query.group_id]
        source_index = positive_query_index[
            (query.group_id, query.source_instruction_id)
        ]
        own_image_rows = [image_index[value] for value in group.positive_ids]
        own_image_set = set(own_image_rows)
        other_image_rows = [
            value for value in all_image_rows if value not in own_image_set
        ]
        other_positive_rows = [
            value
            for value in positive_query_rows
            if query_by_index[value].group_id != query.group_id
        ]
        other_negative_rows = [
            value
            for value in negative_query_rows
            if query_by_index[value].group_id != query.group_id
        ]
        random_image_rows = _sample_indices(
            other_image_rows,
            count=random_samples,
            seed=seed,
            key=f"{query.negative_id}:random-targets",
        )
        random_positive_rows = _sample_indices(
            other_positive_rows,
            count=random_samples,
            seed=seed,
            key=f"{query.negative_id}:random-positive-queries",
        )
        random_negative_rows = _sample_indices(
            other_negative_rows,
            count=random_samples,
            seed=seed,
            key=f"{query.negative_id}:random-negative-queries",
        )

        own_targets = np.asarray(image_embeddings[own_image_rows], dtype=np.float32)
        random_targets = np.asarray(
            image_embeddings[random_image_rows], dtype=np.float32
        )
        source = np.asarray(query_embeddings[source_index], dtype=np.float32)
        negative = np.asarray(query_embeddings[query.query_index], dtype=np.float32)
        source_own_scores = source @ own_targets.T
        negative_own_scores = negative @ own_targets.T
        target_deltas = source_own_scores - negative_own_scores
        source_own = float(source_own_scores.mean())
        negative_own = float(negative_own_scores.mean())
        source_random_target = float((source @ random_targets.T).mean())
        negative_random_target = float((negative @ random_targets.T).mean())
        own_target_mean = own_targets.mean(axis=0)
        random_positive_query_score = float(
            (
                np.asarray(
                    query_embeddings[random_positive_rows], dtype=np.float32
                )
                @ own_target_mean
            ).mean()
        )
        random_negative_query_score = float(
            (
                np.asarray(
                    query_embeddings[random_negative_rows], dtype=np.float32
                )
                @ own_target_mean
            ).mean()
        )
        source_query = query_by_index[source_index]
        rows.append(
            {
                "group_id": query.group_id,
                "instruction_id": query.instruction_id,
                "negative_id": query.negative_id,
                "source_text": source_query.text,
                "negative_text": query.text,
                "semantic_target": query.semantic_target,
                "edit_operation": query.edit_operation,
                "positive_target_count": len(own_image_rows),
                "random_sample_count": random_samples,
                "random_target_ids": "|".join(
                    str(image_ids[index]) for index in random_image_rows
                ),
                "mean_positive_query_own_target_score": source_own,
                "mean_negative_query_own_target_score": negative_own,
                "mean_target_delta": float(target_deltas.mean()),
                "target_raw_win_rate": float(np.mean(target_deltas > 0.0)),
                "target_stable_win_rate": float(
                    np.mean(target_deltas > threshold)
                ),
                "mean_positive_query_random_target_score": source_random_target,
                "positive_query_own_minus_random_target": (
                    source_own - source_random_target
                ),
                "mean_negative_query_random_target_score": negative_random_target,
                "negative_query_own_minus_random_target": (
                    negative_own - negative_random_target
                ),
                "mean_random_positive_query_own_target_score": (
                    random_positive_query_score
                ),
                "positive_query_own_minus_random_positive_query": (
                    source_own - random_positive_query_score
                ),
                "mean_random_negative_query_own_target_score": (
                    random_negative_query_score
                ),
                "negative_query_own_minus_random_negative_query": (
                    negative_own - random_negative_query_score
                ),
                "normal_perturbation_threshold": threshold,
            }
        )
    return pd.DataFrame(rows)


def summarize(
    frame: pd.DataFrame,
    *,
    calibration: dict[str, Any],
    bootstrap_replicates: int,
    seed: int,
) -> pd.DataFrame:
    subset_specs: list[tuple[str, np.ndarray]] = [
        ("all", np.ones(len(frame), dtype=bool)),
        ("strict", (frame["edit_operation"] != "delete").to_numpy()),
        ("add", (frame["edit_operation"] == "add").to_numpy()),
        ("substitute", (frame["edit_operation"] == "substitute").to_numpy()),
        ("delete", (frame["edit_operation"] == "delete").to_numpy()),
    ]
    for semantic in ("object", "attribute", "predicate"):
        subset_specs.append(
            (semantic, (frame["semantic_target"] == semantic).to_numpy())
        )
        subset_specs.append(
            (
                f"strict|{semantic}",
                (
                    (frame["edit_operation"] != "delete")
                    & (frame["semantic_target"] == semantic)
                ).to_numpy(),
            )
        )
    columns = [
        "mean_target_delta",
        "target_raw_win_rate",
        "target_stable_win_rate",
        "mean_positive_query_own_target_score",
        "mean_negative_query_own_target_score",
        "mean_positive_query_random_target_score",
        "positive_query_own_minus_random_target",
        "mean_negative_query_random_target_score",
        "negative_query_own_minus_random_target",
        "mean_random_positive_query_own_target_score",
        "positive_query_own_minus_random_positive_query",
        "mean_random_negative_query_own_target_score",
        "negative_query_own_minus_random_negative_query",
    ]
    rows: list[dict[str, Any]] = []
    for subset_name, mask in subset_specs:
        subset = frame.loc[mask]
        if subset.empty:
            continue
        raw_ci = _bootstrap_hierarchical_mean(
            subset,
            "target_raw_win_rate",
            replicates=bootstrap_replicates,
            seed=seed,
        )
        stable_ci = _bootstrap_hierarchical_mean(
            subset,
            "target_stable_win_rate",
            replicates=bootstrap_replicates,
            seed=seed + 1,
        )
        row: dict[str, Any] = {
            "subset": subset_name,
            "group_count": subset["group_id"].nunique(),
            "instruction_count": subset[["group_id", "instruction_id"]]
            .drop_duplicates()
            .shape[0],
            "negative_count": len(subset),
            "normal_perturbation_threshold": calibration[
                "normal_perturbation_threshold"
            ],
            "threshold_bootstrap_ci_low": calibration["threshold_bootstrap_ci"][0],
            "threshold_bootstrap_ci_high": calibration["threshold_bootstrap_ci"][1],
            "raw_win_rate_ci_low": raw_ci[0],
            "raw_win_rate_ci_high": raw_ci[1],
            "stable_win_rate_ci_low": stable_ci[0],
            "stable_win_rate_ci_high": stable_ci[1],
        }
        row.update({column: _hierarchical_mean(subset, column) for column in columns})
        rows.append(row)
    return pd.DataFrame(rows)


def _cleanup(adapter: Any) -> None:
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
    if args.random_samples <= 0:
        raise ValueError("--random-samples must be positive")
    if args.bootstrap_replicates <= 0:
        raise ValueError("--bootstrap-replicates must be positive")
    checkpoints = parse_checkpoints(args.checkpoint)
    pretrained_clips = parse_pretrained_clips(args.pretrained_clip)
    pretrained_models = parse_pretrained_models(args.pretrained_model)
    adapters = parse_adapters(args.adapter)
    name_groups = [
        {name for name, _ in checkpoints},
        {name for name, _ in pretrained_clips},
        {name for name, _ in pretrained_models},
        {name for name, _ in adapters},
    ]
    duplicate_names = sorted(
        name
        for index, group in enumerate(name_groups)
        for other in name_groups[index + 1 :]
        for name in group & other
    )
    if duplicate_names:
        raise ValueError(f"duplicate model names across inputs: {duplicate_names}")
    models = [
        *((name, path, "checkpoint") for name, path in checkpoints),
        *((name, path, "pretrained") for name, path in pretrained_clips),
        *((name, path, "adapter") for name, path in pretrained_models),
        *((name, path, "adapter") for name, path in adapters),
    ]
    if not models:
        raise ValueError(
            "provide at least one checkpoint, pretrained model, or adapter"
        )
    input_path = args.input.resolve()
    pinpoint_root = args.pinpoint_root.resolve()
    image_cache = pinpoint_root / "image_cache"
    groups, queries, coverage = load_v7_dataset(input_path, limit_groups=None)
    positive_image_ids = sorted({value for group in groups for value in group.positive_ids})
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if str(CIR_PLAY_ROOT) not in sys.path:
        sys.path.insert(0, str(CIR_PLAY_ROOT))
    from run_pinpoint_bge_lincir import (
        choose_device,
        load_or_build_corpus_embeddings,
        make_adapter,
        prepare_hf_runtime_cache,
    )

    device = choose_device(args.device)
    combined: list[pd.DataFrame] = []
    overview_rows: list[dict[str, Any]] = []
    for model_name, model_path, model_kind in models:
        model_dir = output_root / model_name
        cache_root = model_dir / "cache"
        model_dir.mkdir(parents=True, exist_ok=True)
        prepare_hf_runtime_cache(cache_root / "_hf_runtime")
        print(f"[model] {model_name}: {model_path}", flush=True)
        if model_kind == "pretrained":
            adapter = OriginalCLIPAdapter(model_path, device, args.dtype)
        elif model_kind == "adapter":
            adapter = make_adapter(
                model_name,
                model_path,
                device,
                args.dtype,
                hf_cache_dir=(
                    CIR_PLAY_ROOT
                    / "models/checkpoints/lincir/.cache/huggingface"
                ),
                lincir_clip_model_name="large",
            )
        else:
            adapter = TrainedCLIPAdapter(model_path, device, args.dtype)
        image_embeddings, cached_ids = load_or_build_corpus_embeddings(
            adapter,
            positive_image_ids,
            image_cache=image_cache,
            cache_dir=cache_root / "positive_images",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_batches=args.prefetch_batches,
            processor_workers=args.processor_workers,
            processor_prefetch_batches=args.processor_prefetch_batches,
            flush_interval=args.flush_interval,
            force=args.force,
        )
        query_embeddings, _ = load_or_build_query_embeddings(
            adapter,
            queries,
            model_name=model_name,
            model_path=model_path,
            dtype=args.dtype,
            image_cache=image_cache,
            cache_root=cache_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_batches=args.prefetch_batches,
            processor_workers=args.processor_workers,
            processor_prefetch_batches=args.processor_prefetch_batches,
            flush_interval=args.flush_interval,
            force=args.force,
        )
        _cleanup(adapter)
        image_ids = cached_ids.astype(str).tolist()
        image_index = {value: index for index, value in enumerate(image_ids)}
        calibration = calibrate_threshold(
            groups,
            queries,
            query_embeddings,
            image_embeddings,
            image_index,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
        )
        calibration.update(
            {
                "model_name": model_name,
                "checkpoint": (
                    str(model_path) if model_kind == "checkpoint" else None
                ),
                "model_path": str(model_path),
                "model_kind": model_kind,
                "random_sample_count": args.random_samples,
                "seed": args.seed,
                "coverage": coverage,
            }
        )
        scores = score_negatives(
            groups,
            queries,
            query_embeddings,
            image_embeddings,
            image_ids,
            threshold=float(calibration["normal_perturbation_threshold"]),
            random_samples=args.random_samples,
            seed=args.seed,
        )
        summary = summarize(
            scores,
            calibration=calibration,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
        )
        scores.to_csv(model_dir / "per_negative_random10.csv", index=False)
        summary.to_csv(model_dir / "summary.csv", index=False)
        write_json_atomic(model_dir / "calibration.json", calibration)
        combined_frame = summary.copy()
        combined_frame.insert(0, "model", model_name)
        combined.append(combined_frame)
        for subset in ("all", "strict"):
            row = combined_frame.loc[combined_frame["subset"] == subset]
            if not row.empty:
                overview_rows.append(row.iloc[0].to_dict())
        print(
            f"[done] {model_name} threshold="
            f"{calibration['normal_perturbation_threshold']:.6f}",
            flush=True,
        )

    pd.concat(combined, ignore_index=True).to_csv(
        output_root / "combined_summary.csv", index=False
    )
    pd.DataFrame(overview_rows).to_csv(output_root / "overview_all_strict.csv", index=False)
    write_json_atomic(
        output_root / "run_metadata.json",
        {
            "input": str(input_path),
            "models": [name for name, _, _ in models],
            "positive_image_count": len(positive_image_ids),
            "query_count": len(queries),
            "random_sample_count": args.random_samples,
            "random_sampling": (
                "deterministic without replacement; excludes the current PinPoint "
                "group; identical IDs are used across models"
            ),
            "seed": args.seed,
        },
    )
    print(f"[done] combined={output_root / 'overview_all_strict.csv'}")


if __name__ == "__main__":
    main()
