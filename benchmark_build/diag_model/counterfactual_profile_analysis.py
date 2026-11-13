#!/usr/bin/env python3
"""Sparse counterfactual-response diagnostics on official PinPoint retrieval.

Official PinPoint supplies the retrieval corpus, positive/negative image labels,
and the headline mAP@10/NegRecall@10 protocol.  PinPoint V7 supplies accepted
counterfactual text edits for a 500-group subset of those official queries.
The two sources are joined by official query ID; V7 is never treated as a
separate retrieval benchmark.

The primary response does not require a complete six-cell edit grid.  For each
candidate, edits are averaged within category and then equally across whatever
accepted ADD/SUBSTITUTE categories are available for that query.  Missing cells
remain NaN.  Fixed-cell and masked-grid probes are secondary diagnostics.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import write_json_atomic
from pinpoint_full_diagnostics import (
    GroupSpec,
    QuerySpec,
    load_or_build_query_embeddings,
    load_v7_dataset,
    query_fingerprint,
)
from run_clip_pinpoint_official import load_official_queries, normalize_query_id
from trained_clip_adapter import TRAINED_CLIP_MODELS, TrainedCLIPAdapter


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]
PINPOINT_ROOT = WORKSPACE_ROOT / "pinpoint-dataset"
DEFAULT_INPUT = (
    BENCHMARK_ROOT
    / "cir_qdiag_openai"
    / "outputs"
    / "pinpoint_pilot_500_v7"
    / "final.jsonl"
)
DEFAULT_OUTPUT = (
    BENCHMARK_ROOT
    / "cir_qdiag_openai"
    / "outputs"
    / "pinpoint_pilot_500_v7"
    / "counterfactual_candidate_profiles"
)
OFFICIAL_OUTPUT_ROOT = (
    BENCHMARK_ROOT / "cir_qdiag_openai" / "outputs" / "pinpoint_official_full_lora"
)
DEFAULT_OFFICIAL_CACHES = {
    name: OFFICIAL_OUTPUT_ROOT / name
    for name in (
        "CLIP-L-full-baseline",
        "CLIP-L-full-hard-w0.5",
        "CLIP-L-lora-r16-baseline",
        "CLIP-L-lora-r16-hard-w0.5",
    )
}
DEFAULT_OFFICIAL_CACHES.update(
    {
        name: BENCHMARK_ROOT
        / "cir_qdiag_openai"
        / "outputs"
        / "pinpoint_official_full"
        / name
        for name in ("CLIP-L-baseline", "CLIP-L-hard-w0.5")
    }
)
DEFAULT_MODELS = "CLIP-L-full-baseline,CLIP-L-full-hard-w0.5"
FROZEN_CLIP_CORPUS_CACHE = (
    BENCHMARK_ROOT
    / "cir_qdiag_openai"
    / "outputs"
    / "pinpoint_pilot_500_v7"
    / "clip_l_pinpoint_diagnostics"
    / "_shared_frozen_clip_l_corpus"
)
RESPONSE_CATEGORIES = (
    "object_add",
    "object_substitute",
    "attribute_add",
    "attribute_substitute",
    "predicate_add",
    "predicate_substitute",
)
RESPONSE_COLUMNS = tuple(f"d_{value}" for value in RESPONSE_CATEGORIES)
MASK_COLUMNS = tuple(f"has_{value}" for value in RESPONSE_CATEGORIES)
PERMUTED_RESPONSE_COLUMNS = tuple(
    f"permuted_d_{value}" for value in RESPONSE_CATEGORIES
)
PERMUTED_MASK_COLUMNS = tuple(f"permuted_has_{value}" for value in RESPONSE_CATEGORIES)
PROFILE_DIMENSIONS = (
    "d_available",
    "d_edit_micro",
    "d_edit_min",
    "d_edit_max",
    "d_edit_std",
    "d_min",
    "d_max",
    "d_std",
    "d_object",
    "d_attribute",
    "d_predicate",
    "d_add",
    "d_substitute",
    "d_available_minus_abs_paraphrase",
    *RESPONSE_COLUMNS,
)
PRIMARY_PROBE_FEATURE = "source_plus_d_available"
PRIMARY_PROBE_SCOPE = "pool_nonpositive_benchmark"
RANKING_JACCARD_KS = (1, 5, 10, 20, 50)
RERANK_PROBE_SCOPES = ("native_clean", "pool_nonpositive_benchmark")
RANK_VIOLATION_PROBE_SCOPES = (
    "rank_violating_official_negative",
    "rank_violating_pool_nonpositive",
)
# These scopes have a model-independent candidate/label cohort once both models
# are evaluated on the same shared-union pool.  Rank-violating scopes are
# deliberately excluded: eligibility there depends on each model's native score.
CROSS_MODEL_FIXED_PROBE_SCOPES = (
    "native_clean",
    "pool_nonpositive_benchmark",
    "augmented_clean",
)
RANK_VIOLATION_FEATURES = {
    "source_similarity_strict",
    "d_available_only",
    "source_plus_d_available",
    "mask_only",
    "imputed_6d_values_only",
    "masked_6d",
    "source_plus_masked_6d",
}
FEATURE_COMPARISONS = (
    ("source_plus_d_available", "source_similarity_strict"),
    ("ensemble_plus_d_group", "positive_ensemble_strict"),
    ("source_paraphrase_and_d", "source_plus_paraphrase_strict"),
    ("response_moments", "source_similarity_strict"),
    ("positive_ensemble_strict", "source_similarity_strict"),
    ("imputed_6d_values_only", "mask_only"),
    ("masked_6d", "mask_only"),
    ("masked_6d", "imputed_6d_values_only"),
    ("source_plus_masked_6d", "source_similarity_strict"),
    ("source_plus_masked_6d", "source_plus_category_permuted_6d"),
    ("source_plus_d_available", "candidate_shuffled_placebo"),
    ("source_plus_object_pair_2d", "source_similarity_object_pair"),
    ("positive_ensemble_object_pair", "source_similarity_object_pair"),
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    checkpoint: Path
    official_cache: Path
    corpus_cache: Path | None = None
    allow_shared_frozen_corpus: bool = False


@dataclass
class ModelArtifacts:
    spec: ModelSpec
    corpus: np.ndarray
    signatures: np.ndarray
    signature_to_index: dict[str, int]
    positive_queries: np.ndarray
    query_keys: np.ndarray
    key_to_index: dict[str, int]
    top_scores: np.ndarray
    top_indices: np.ndarray


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    cohort: str
    columns: tuple[str, ...]
    impute: bool = False


@dataclass
class ProbeOutput:
    result: dict[str, Any]
    clean_predictions: pd.DataFrame
    pool_predictions: pd.DataFrame


class ProbeNotEvaluableError(ValueError):
    """Raised when a selected sparse cohort cannot support grouped OOF."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=PINPOINT_ROOT / "pinpoint_licensed.parquet",
    )
    parser.add_argument("--pinpoint-root", type=Path, default=PINPOINT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Override/register a trained checkpoint; repeat as needed.",
    )
    parser.add_argument(
        "--official-cache",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Override/register the matching official PinPoint model output directory.",
    )
    parser.add_argument(
        "--comparison",
        action="append",
        default=[],
        metavar="BASE,HARD",
        help="Paired model comparison on a fixed union of their native Top-K pools.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--prefetch-batches", type=int, default=4)
    parser.add_argument("--processor-workers", type=int, default=2)
    parser.add_argument("--processor-prefetch-batches", type=int, default=2)
    parser.add_argument("--flush-interval", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-groups", type=int, default=None)
    parser.add_argument("--force-negative-queries", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate data/checkpoint/cache identity without encoding or analysis.",
    )
    parser.add_argument(
        "--ranking-jaccard-only",
        action="store_true",
        help="Compute native baseline/hard Top-K Jaccard tables and exit.",
    )
    parser.add_argument(
        "--cross-model-rerank-only",
        action="store_true",
        help=(
            "Rerank each model's native Top-50 with the paired model's normal "
            "positive-query similarity and exit; never uses negative text."
        ),
    )
    probe_group = parser.add_mutually_exclusive_group()
    probe_group.add_argument(
        "--enable-oof-probes",
        dest="skip_probes",
        action="store_false",
        help=(
            "Opt in to the legacy grouped-OOF LogisticRegression diagnostics. "
            "The default simple protocol uses no learned probe or learned reranker."
        ),
    )
    probe_group.add_argument(
        "--skip-probes",
        dest="skip_probes",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(skip_probes=True)
    parser.add_argument("--skip-rerank", action="store_true")
    parser.add_argument(
        "--save-shared-profiles",
        action="store_true",
        help="Persist large candidate-level shared-union parquet files.",
    )
    parser.add_argument("--rerank-depths", default="10,20,50")
    parser.add_argument(
        "--official-eval-all",
        action="store_true",
        help=(
            "Invoke the original evaluator for every rerank variant. By default it "
            "is invoked for native retrieval and pre-registered primary variants; "
            "all variants still receive mathematically identical unrounded metrics."
        ),
    )
    return parser.parse_args()


def parse_named_paths(values: Sequence[str], *, option: str) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} expects NAME=PATH, got {value!r}")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        raw_path = raw_path.strip()
        if not name or not raw_path:
            raise ValueError(f"{option} expects NAME=PATH, got {value!r}")
        if name in parsed:
            raise ValueError(f"duplicate {option} registration for {name}")
        parsed[name] = Path(raw_path).resolve()
    return parsed


def resolve_model_specs(args: argparse.Namespace) -> list[ModelSpec]:
    names = list(
        dict.fromkeys(
            value.strip() for value in args.models.split(",") if value.strip()
        )
    )
    checkpoints = {
        name: Path(path).resolve() for name, path in TRAINED_CLIP_MODELS.items()
    }
    checkpoints.update(parse_named_paths(args.checkpoint, option="--checkpoint"))
    caches = {name: path.resolve() for name, path in DEFAULT_OFFICIAL_CACHES.items()}
    caches.update(parse_named_paths(args.official_cache, option="--official-cache"))
    unknown_checkpoints = sorted(set(names) - set(checkpoints))
    unknown_caches = sorted(set(names) - set(caches))
    if unknown_checkpoints:
        raise ValueError(f"models missing checkpoints: {unknown_checkpoints}")
    if unknown_caches:
        raise ValueError(f"models missing official caches: {unknown_caches}")
    specs = [
        ModelSpec(
            name,
            checkpoints[name],
            caches[name],
            corpus_cache=FROZEN_CLIP_CORPUS_CACHE.resolve()
            if name in {"CLIP-L-baseline", "CLIP-L-hard-w0.5"}
            else None,
            allow_shared_frozen_corpus=name in {"CLIP-L-baseline", "CLIP-L-hard-w0.5"},
        )
        for name in names
    ]
    for spec in specs:
        if not spec.checkpoint.is_file():
            raise FileNotFoundError(spec.checkpoint)
        if not spec.official_cache.is_dir():
            raise FileNotFoundError(spec.official_cache)
        if spec.corpus_cache is not None and not spec.corpus_cache.is_dir():
            raise FileNotFoundError(spec.corpus_cache)
    return specs


def parse_comparisons(
    values: Sequence[str],
    model_names: Sequence[str],
) -> list[tuple[str, str]]:
    if not values and {
        "CLIP-L-full-baseline",
        "CLIP-L-full-hard-w0.5",
    }.issubset(model_names):
        values = ["CLIP-L-full-baseline,CLIP-L-full-hard-w0.5"]
    pairs: list[tuple[str, str]] = []
    for value in values:
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if len(parts) != 2:
            raise ValueError(f"--comparison expects BASE,HARD, got {value!r}")
        if any(part not in model_names for part in parts):
            raise ValueError(f"comparison contains an unselected model: {parts}")
        pairs.append((parts[0], parts[1]))
    return list(dict.fromkeys(pairs))


def _query_cache_candidates(
    model_dir: Path,
    *,
    expected_keys: set[str],
) -> list[Path]:
    candidates = []
    for metadata_path in sorted(
        (model_dir / "cache" / "queries").glob("*/query_metadata.json")
    ):
        keys_path = metadata_path.with_name("query_keys.npy")
        embeddings_path = metadata_path.with_name("query_embeddings.npy")
        if not keys_path.is_file() or not embeddings_path.is_file():
            continue
        keys = np.load(keys_path, mmap_mode="r", allow_pickle=False).astype(str)
        key_values = keys.tolist()
        if (
            len(key_values) == len(expected_keys)
            and len(key_values) == len(set(key_values))
            and set(key_values) == expected_keys
        ):
            candidates.append(metadata_path.parent)
    return candidates


def _matching_resolved_path(first: str | Path, second: str | Path) -> bool:
    return Path(first).resolve() == Path(second).resolve()


def load_model_artifacts(
    spec: ModelSpec,
    *,
    top_k: int,
    ground_truth_path: Path,
) -> ModelArtifacts:
    _, official_queries = load_official_queries(ground_truth_path)
    expected_query_keys = {query.query_key for query in official_queries}
    corpus_dir = spec.corpus_cache or spec.official_cache / "cache" / "corpus"
    corpus_path = corpus_dir / "corpus_embeddings.npy"
    signatures_path = corpus_dir / "corpus_signatures.npy"
    corpus_metadata_path = corpus_dir / "corpus_metadata.json"
    for required in (corpus_path, signatures_path, corpus_metadata_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    corpus_metadata = json.loads(corpus_metadata_path.read_text(encoding="utf-8"))
    if (
        not _matching_resolved_path(corpus_metadata["model_path"], spec.checkpoint)
        and not spec.allow_shared_frozen_corpus
    ):
        raise ValueError(
            f"{spec.name} corpus cache belongs to {corpus_metadata['model_path']}, "
            f"not {spec.checkpoint}"
        )
    corpus = np.load(corpus_path, mmap_mode="r")
    signatures = np.load(signatures_path, mmap_mode="r", allow_pickle=False).astype(str)
    if corpus.ndim != 2 or len(corpus) != len(signatures):
        raise ValueError(f"invalid corpus cache for {spec.name}: {corpus.shape}")
    if (
        int(corpus_metadata.get("num_images", -1)) != len(corpus)
        or int(corpus_metadata.get("embedding_dim", -1)) != corpus.shape[1]
    ):
        raise ValueError(f"corpus metadata shape mismatch for {spec.name}")
    if len(signatures) != len(set(signatures.tolist())):
        raise ValueError(f"duplicate corpus signatures for {spec.name}")

    query_dirs = _query_cache_candidates(
        spec.official_cache,
        expected_keys=expected_query_keys,
    )
    matching_query_dirs = []
    for query_dir in query_dirs:
        metadata = json.loads(
            (query_dir / "query_metadata.json").read_text(encoding="utf-8")
        )
        expected_fingerprint = query_fingerprint(
            official_queries,
            model_name=spec.name,
            model_path=spec.checkpoint,
            dtype=str(metadata.get("dtype", "")),
        )
        if (
            _matching_resolved_path(metadata["model_path"], spec.checkpoint)
            and metadata.get("model_name") == spec.name
            and metadata.get("fingerprint") == expected_fingerprint
        ):
            matching_query_dirs.append(query_dir)
    if len(matching_query_dirs) != 1:
        raise ValueError(
            f"expected one matching official query cache for {spec.name}, "
            f"found {matching_query_dirs}"
        )
    query_dir = matching_query_dirs[0]
    positive_queries = np.load(query_dir / "query_embeddings.npy", mmap_mode="r")
    query_keys = np.load(
        query_dir / "query_keys.npy", mmap_mode="r", allow_pickle=False
    ).astype(str)
    if positive_queries.ndim != 2 or len(positive_queries) != len(query_keys):
        raise ValueError(f"invalid official query cache for {spec.name}")
    if positive_queries.shape[1] != corpus.shape[1]:
        raise ValueError(f"query/corpus embedding dimension mismatch for {spec.name}")
    query_key_values = query_keys.tolist()
    if (
        len(query_key_values) != len(set(query_key_values))
        or set(query_key_values) != expected_query_keys
    ):
        raise ValueError(f"official query identifiers mismatch for {spec.name}")
    checkpoint_mtime = spec.checkpoint.stat().st_mtime_ns
    if checkpoint_mtime > min(
        corpus_path.stat().st_mtime_ns,
        (query_dir / "query_embeddings.npy").stat().st_mtime_ns,
    ):
        raise ValueError(
            f"{spec.name} checkpoint is newer than its official embedding cache; "
            "rebuild the official cache before analysis"
        )

    query_metadata = json.loads(
        (query_dir / "query_metadata.json").read_text(encoding="utf-8")
    )
    search_fingerprint = stable_hash(
        {
            "protocol_version": 1,
            "query_fingerprint": query_metadata["fingerprint"],
            "corpus_signature_hash": stable_hash(signatures.tolist()),
            "top_k": top_k,
            "method": "exact_inner_product_torch",
        }
    )
    search_path = (
        spec.official_cache / "cache" / "search" / f"topk_{search_fingerprint[:16]}.npz"
    )
    if not search_path.is_file():
        available = sorted(
            str(path)
            for path in (spec.official_cache / "cache" / "search").glob("topk_*.npz")
        )
        raise FileNotFoundError(
            f"exact Top-{top_k} cache for {spec.name} is absent; available={available}"
        )
    try:
        with np.load(search_path, allow_pickle=False) as archive:
            if set(archive.files) != {"scores", "indices"}:
                raise ValueError(f"invalid search archive members: {archive.files}")
            top_scores = archive["scores"]
            top_indices = archive["indices"]
    except Exception as exc:
        raise ValueError(f"failed to read search cache {search_path}: {exc}") from exc
    expected_shape = (len(positive_queries), top_k)
    if top_scores.shape != expected_shape or top_indices.shape != expected_shape:
        raise ValueError(
            f"invalid Top-{top_k} shape for {spec.name}: "
            f"{top_scores.shape}/{top_indices.shape}"
        )
    if (
        not np.isfinite(top_scores).all()
        or top_indices.min() < 0
        or top_indices.max() >= len(corpus)
        or np.any(np.diff(top_scores, axis=1) > 1e-6)
        or np.any(np.diff(np.sort(top_indices, axis=1), axis=1) == 0)
    ):
        raise ValueError(f"invalid scores/indices in {search_path}")

    sample_rows = np.linspace(
        0,
        len(positive_queries) - 1,
        num=min(5, len(positive_queries)),
        dtype=int,
    )
    for row in sample_rows:
        recomputed = (
            np.asarray(positive_queries[row], dtype=np.float32)
            @ np.asarray(corpus[top_indices[row]], dtype=np.float32).T
        )
        if not np.allclose(recomputed, top_scores[row], atol=2e-4, rtol=2e-4):
            raise ValueError(f"Top-K scores do not match embeddings for {spec.name}")

    return ModelArtifacts(
        spec=spec,
        corpus=corpus,
        signatures=signatures,
        signature_to_index={value: index for index, value in enumerate(signatures)},
        positive_queries=positive_queries,
        query_keys=query_keys,
        key_to_index={value: index for index, value in enumerate(query_keys)},
        top_scores=top_scores,
        top_indices=top_indices,
    )


def official_key(query_id: str) -> str:
    return f"official::{normalize_query_id(query_id)}"


def clean_identifier_values(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, list | tuple):
        return ()
    return tuple(
        dict.fromkeys(
            str(value)
            for value in values
            if value is not None
            and str(value).strip().lower() not in {"", "none", "nan"}
        )
    )


def response_category(query: QuerySpec) -> str:
    return f"{query.semantic_target}_{query.edit_operation}"


def strict_negative_queries(queries: Sequence[QuerySpec]) -> list[QuerySpec]:
    selected: list[QuerySpec] = []
    seen_negative_ids: set[str] = set()
    for query in queries:
        if query.query_kind != "negative" or query.edit_operation == "delete":
            continue
        if query.negative_id in seen_negative_ids:
            raise ValueError(f"duplicate accepted negative_id: {query.negative_id}")
        seen_negative_ids.add(query.negative_id)
        selected.append(query)
    return selected


def validated_positive_ids_by_group(input_path: Path) -> dict[str, set[str]]:
    values: dict[str, set[str]] = {}
    with input_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            values[str(record["group_id"])] = {
                str(value)
                for value in record.get("metadata", {}).get(
                    "validated_positive_image_ids", []
                )
                if value
            }
    return values


def make_group_fold_map(
    group_ids: Iterable[str],
    *,
    folds: int,
    seed: int,
) -> dict[str, int]:
    unique = np.asarray(sorted(set(group_ids)), dtype=object)
    if folds < 2 or folds > len(unique):
        raise ValueError(f"folds must be in [2, {len(unique)}], got {folds}")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    return {str(group): index % folds for index, group in enumerate(unique)}


def make_balanced_group_fold_map(
    coverage_frame: pd.DataFrame,
    *,
    folds: int,
    seed: int,
) -> dict[str, int]:
    """Greedily balance all/strict/object-pair query loads across V7 groups."""

    aggregate = (
        coverage_frame.groupby("group_id", sort=True)
        .agg(
            query_count=("query_id", "size"),
            strict_query_count=("has_any_strict", "sum"),
            object_pair_query_count=("has_object_pair", "sum"),
        )
        .reset_index()
    )
    if folds < 2 or folds > len(aggregate):
        raise ValueError(f"folds must be in [2, {len(aggregate)}], got {folds}")
    rng = np.random.default_rng(seed)
    aggregate["tie_break"] = rng.random(len(aggregate))
    aggregate = aggregate.sort_values(
        [
            "object_pair_query_count",
            "strict_query_count",
            "query_count",
            "tie_break",
        ],
        ascending=[False, False, False, True],
    )
    columns = ["query_count", "strict_query_count", "object_pair_query_count"]
    total = aggregate[columns].sum().to_numpy(dtype=np.float64)
    targets = np.maximum(total / folds, 1.0)
    loads = np.zeros((folds, len(columns)), dtype=np.float64)
    group_loads = np.zeros(folds, dtype=np.int64)
    fold_tie_order = rng.permutation(folds)
    mapping: dict[str, int] = {}
    for row_number, row in enumerate(aggregate.itertuples(index=False)):
        vector = np.asarray(
            [
                row.query_count,
                row.strict_query_count,
                row.object_pair_query_count,
            ],
            dtype=np.float64,
        )
        if row_number < folds:
            selected_fold = int(fold_tie_order[row_number])
        else:
            candidate_scores = []
            for fold in range(folds):
                proposed = loads.copy()
                proposed[fold] += vector
                imbalance = float(np.sum((proposed / targets) ** 2))
                candidate_scores.append(
                    (
                        imbalance,
                        int(group_loads[fold]),
                        int(np.where(fold_tie_order == fold)[0][0]),
                        fold,
                    )
                )
            selected_fold = min(candidate_scores)[-1]
        mapping[str(row.group_id)] = selected_fold
        loads[selected_fold] += vector
        group_loads[selected_fold] += 1
    return mapping


def candidate_shuffle_within_query(
    frame: pd.DataFrame,
    column: str,
    *,
    seed: int,
    strata: Sequence[str] = (),
) -> pd.Series:
    shuffled = frame[column].copy()
    rng = np.random.default_rng(seed)
    group_columns = ["query_id", *strata]
    grouper: str | list[str] = (
        group_columns[0] if len(group_columns) == 1 else group_columns
    )
    for _, indices in frame.groupby(grouper, sort=False).groups.items():
        values = shuffled.loc[indices].to_numpy(copy=True)
        rng.shuffle(values)
        shuffled.loc[indices] = values
    return shuffled


def build_coverage_frame(
    groups: Sequence[GroupSpec],
    positive_queries: Sequence[QuerySpec],
    negatives: Sequence[QuerySpec],
) -> pd.DataFrame:
    categories_by_query: dict[str, Counter[str]] = defaultdict(Counter)
    for query in negatives:
        categories_by_query[query.instruction_id][response_category(query)] += 1
    group_by_query = {
        instruction_id: group.group_id
        for group in groups
        for instruction_id in group.instruction_ids
    }
    rows = []
    for query in positive_queries:
        counts = categories_by_query[query.instruction_id]
        row: dict[str, Any] = {
            "group_id": group_by_query[query.instruction_id],
            "query_id": query.instruction_id,
            "strict_edit_count": sum(counts.values()),
            "available_category_count": sum(
                counts[value] > 0 for value in RESPONSE_CATEGORIES
            ),
            "has_any_strict": bool(counts),
            "has_object_pair": bool(
                counts["object_add"] and counts["object_substitute"]
            ),
            "has_complete_6d": all(counts[value] > 0 for value in RESPONSE_CATEGORIES),
        }
        for category in RESPONSE_CATEGORIES:
            row[f"count_{category}"] = counts[category]
            row[f"has_{category}"] = bool(counts[category])
        rows.append(row)
    return pd.DataFrame(rows)


def write_coverage_outputs(
    coverage_frame: pd.DataFrame,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage_frame.to_csv(
        output_dir / "negative_text_coverage_by_query.csv", index=False
    )
    dimension_rows = []
    for category in RESPONSE_CATEGORIES:
        mask = coverage_frame[f"has_{category}"]
        dimension_rows.append(
            {
                "dimension": category,
                "query_count": int(mask.sum()),
                "group_count": int(coverage_frame.loc[mask, "group_id"].nunique()),
                "negative_text_count": int(coverage_frame[f"count_{category}"].sum()),
            }
        )
    dimension_rows.extend(
        [
            {
                "dimension": "any_strict",
                "query_count": int(coverage_frame["has_any_strict"].sum()),
                "group_count": int(
                    coverage_frame.loc[
                        coverage_frame["has_any_strict"], "group_id"
                    ].nunique()
                ),
                "negative_text_count": int(coverage_frame["strict_edit_count"].sum()),
            },
            {
                "dimension": "object_add+object_substitute",
                "query_count": int(coverage_frame["has_object_pair"].sum()),
                "group_count": int(
                    coverage_frame.loc[
                        coverage_frame["has_object_pair"], "group_id"
                    ].nunique()
                ),
                "negative_text_count": None,
            },
            {
                "dimension": "complete_6d",
                "query_count": int(coverage_frame["has_complete_6d"].sum()),
                "group_count": int(
                    coverage_frame.loc[
                        coverage_frame["has_complete_6d"], "group_id"
                    ].nunique()
                ),
                "negative_text_count": None,
            },
        ]
    )
    dimension_frame = pd.DataFrame(dimension_rows)
    dimension_frame.to_csv(
        output_dir / "negative_text_coverage_summary.csv", index=False
    )
    summary = {
        "query_count": len(coverage_frame),
        "group_count": int(coverage_frame["group_id"].nunique()),
        "query_with_any_strict_count": int(coverage_frame["has_any_strict"].sum()),
        "query_without_strict_count": int((~coverage_frame["has_any_strict"]).sum()),
        "complete_6d_query_count": int(coverage_frame["has_complete_6d"].sum()),
        "available_category_histogram": {
            str(key): int(value)
            for key, value in coverage_frame["available_category_count"]
            .value_counts()
            .sort_index()
            .items()
        },
        "dimensions": dimension_rows,
    }
    write_json_atomic(output_dir / "negative_text_coverage_summary.json", summary)
    return summary


def encode_negative_queries(
    artifact: ModelArtifacts,
    negatives: Sequence[QuerySpec],
    *,
    output_dir: Path,
    image_cache: Path,
    device: str,
    dtype: str,
    batch_size: int,
    num_workers: int,
    prefetch_batches: int,
    processor_workers: int,
    processor_prefetch_batches: int,
    flush_interval: int,
    force: bool,
    analysis_only: bool,
) -> np.ndarray:
    cache_root = output_dir / artifact.spec.name / "negative_query_cache"
    if analysis_only:
        candidates = []
        for metadata_path in cache_root.glob("queries/*/query_metadata.json"):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            keys_path = metadata_path.with_name("query_keys.npy")
            embeddings_path = metadata_path.with_name("query_embeddings.npy")
            if not keys_path.is_file() or not embeddings_path.is_file():
                continue
            keys = np.load(keys_path, allow_pickle=False).astype(str).tolist()
            expected_fingerprint = query_fingerprint(
                negatives,
                model_name=f"{artifact.spec.name}::strict-negative-profile",
                model_path=artifact.spec.checkpoint,
                dtype=str(metadata.get("dtype", "")),
            )
            if (
                keys == [query.query_key for query in negatives]
                and _matching_resolved_path(
                    metadata.get("model_path", ""), artifact.spec.checkpoint
                )
                and metadata.get("fingerprint") == expected_fingerprint
                and metadata.get("dtype") == dtype
            ):
                candidates.append(embeddings_path)
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"analysis-only expected one negative-query cache for "
                f"{artifact.spec.name}, found {candidates}"
            )
        embeddings = np.load(candidates[0], mmap_mode="r")
        if embeddings.shape != (len(negatives), artifact.corpus.shape[1]):
            raise ValueError(
                f"invalid negative-query cache shape for {artifact.spec.name}: "
                f"{embeddings.shape}"
            )
        return embeddings

    import torch

    adapter = TrainedCLIPAdapter(
        artifact.spec.checkpoint,
        torch.device(device),
        dtype,
    )
    embeddings, _ = load_or_build_query_embeddings(
        adapter,
        negatives,
        model_name=f"{artifact.spec.name}::strict-negative-profile",
        model_path=artifact.spec.checkpoint,
        dtype=dtype,
        image_cache=image_cache,
        cache_root=cache_root,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_batches=prefetch_batches,
        processor_workers=processor_workers,
        processor_prefetch_batches=processor_prefetch_batches,
        flush_interval=flush_interval,
        force=force,
    )
    del adapter
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if embeddings.shape != (len(negatives), artifact.corpus.shape[1]):
        raise ValueError(
            f"negative-query embedding shape mismatch for {artifact.spec.name}: "
            f"{embeddings.shape}"
        )
    return embeddings


def _finite_mean(values: np.ndarray, axis: int = 1) -> np.ndarray:
    counts = np.isfinite(values).sum(axis=axis)
    totals = np.nansum(values, axis=axis)
    output = np.full(counts.shape, np.nan, dtype=np.float32)
    np.divide(totals, counts, out=output, where=counts > 0)
    return output


def _operation_or_semantic_mean(
    category_values: dict[str, np.ndarray],
    categories: Sequence[str],
    candidate_count: int,
) -> np.ndarray:
    values = [
        category_values[value] for value in categories if value in category_values
    ]
    if not values:
        return np.full(candidate_count, np.nan, dtype=np.float32)
    return np.mean(np.stack(values, axis=1), axis=1, dtype=np.float32)


def summarize_candidate_responses(
    positive_query: np.ndarray,
    candidate_embeddings: np.ndarray,
    negative_embeddings: np.ndarray,
    negative_categories: Sequence[str],
) -> dict[str, np.ndarray]:
    candidates = np.asarray(candidate_embeddings, dtype=np.float32)
    positive = np.asarray(positive_query, dtype=np.float32)
    negatives = np.asarray(negative_embeddings, dtype=np.float32)
    sim_positive = candidates @ positive
    candidate_count = len(candidates)
    if len(negatives) == 0:
        empty = np.full(candidate_count, np.nan, dtype=np.float32)
        result = {
            "sim_positive": sim_positive.astype(np.float32, copy=False),
            "d_edit_micro": empty.copy(),
            "d_edit_min": empty.copy(),
            "d_edit_max": empty.copy(),
            "d_edit_std": empty.copy(),
            "d_available": empty.copy(),
            "d_min": empty.copy(),
            "d_max": empty.copy(),
            "d_std": empty.copy(),
            "d_object": empty.copy(),
            "d_attribute": empty.copy(),
            "d_predicate": empty.copy(),
            "d_add": empty.copy(),
            "d_substitute": empty.copy(),
        }
        result.update({column: empty.copy() for column in RESPONSE_COLUMNS})
        return result

    negative_scores = candidates @ negatives.T
    edit_deltas = sim_positive[:, None] - negative_scores
    category_values: dict[str, np.ndarray] = {}
    for category in RESPONSE_CATEGORIES:
        positions = [
            index
            for index, value in enumerate(negative_categories)
            if value == category
        ]
        if positions:
            category_values[category] = np.mean(
                edit_deltas[:, positions], axis=1, dtype=np.float32
            )
    category_matrix = np.full(
        (candidate_count, len(RESPONSE_CATEGORIES)),
        np.nan,
        dtype=np.float32,
    )
    for index, category in enumerate(RESPONSE_CATEGORIES):
        if category in category_values:
            category_matrix[:, index] = category_values[category]
    result = {
        "sim_positive": sim_positive.astype(np.float32, copy=False),
        "d_edit_micro": np.mean(edit_deltas, axis=1, dtype=np.float32),
        "d_edit_min": np.min(edit_deltas, axis=1),
        "d_edit_max": np.max(edit_deltas, axis=1),
        "d_edit_std": np.std(edit_deltas, axis=1),
        "d_available": _finite_mean(category_matrix),
        "d_min": np.nanmin(category_matrix, axis=1),
        "d_max": np.nanmax(category_matrix, axis=1),
        "d_std": np.nanstd(category_matrix, axis=1),
        "d_object": _operation_or_semantic_mean(
            category_values,
            ("object_add", "object_substitute"),
            candidate_count,
        ),
        "d_attribute": _operation_or_semantic_mean(
            category_values,
            ("attribute_add", "attribute_substitute"),
            candidate_count,
        ),
        "d_predicate": _operation_or_semantic_mean(
            category_values,
            ("predicate_add", "predicate_substitute"),
            candidate_count,
        ),
        "d_add": _operation_or_semantic_mean(
            category_values,
            ("object_add", "attribute_add", "predicate_add"),
            candidate_count,
        ),
        "d_substitute": _operation_or_semantic_mean(
            category_values,
            (
                "object_substitute",
                "attribute_substitute",
                "predicate_substitute",
            ),
            candidate_count,
        ),
    }
    for category, column in zip(RESPONSE_CATEGORIES, RESPONSE_COLUMNS, strict=True):
        result[column] = category_matrix[:, RESPONSE_CATEGORIES.index(category)]
    return result


def summarize_positive_paraphrases(
    positive_query: np.ndarray,
    group_positive_queries: np.ndarray,
    candidate_embeddings: np.ndarray,
    *,
    source_position: int,
) -> dict[str, np.ndarray]:
    candidates = np.asarray(candidate_embeddings, dtype=np.float32)
    scores = candidates @ np.asarray(group_positive_queries, dtype=np.float32).T
    source_scores = candidates @ np.asarray(positive_query, dtype=np.float32)
    ensemble = np.mean(scores, axis=1, dtype=np.float32)
    other_positions = [
        index for index in range(scores.shape[1]) if index != source_position
    ]
    if other_positions:
        deltas = source_scores[:, None] - scores[:, other_positions]
        paraphrase_mean = np.mean(deltas, axis=1, dtype=np.float32)
        paraphrase_std = np.std(deltas, axis=1)
    else:
        paraphrase_mean = np.full(len(candidates), np.nan, dtype=np.float32)
        paraphrase_std = np.full(len(candidates), np.nan, dtype=np.float32)
    return {
        "sim_positive_ensemble": ensemble,
        "paraphrase_delta_mean": paraphrase_mean,
        "paraphrase_delta_std": paraphrase_std,
    }


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prepare_ground_truth(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    required = {"query_id", "positive_candidates", "negative_candidates"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"ground truth is missing columns: {missing}")
    frame["normalized_query_id"] = frame["query_id"].map(normalize_query_id)
    if frame["normalized_query_id"].duplicated().any():
        duplicates = frame.loc[
            frame["normalized_query_id"].duplicated(), "normalized_query_id"
        ].tolist()
        raise ValueError(f"duplicate normalized query IDs: {duplicates[:5]}")
    frame["positive_clean"] = frame["positive_candidates"].map(clean_identifier_values)
    frame["negative_clean"] = frame["negative_candidates"].map(clean_identifier_values)
    return frame.set_index("normalized_query_id", drop=False)


def _query_negative_positions(
    negatives: Sequence[QuerySpec],
) -> dict[str, list[int]]:
    positions: dict[str, list[int]] = defaultdict(list)
    for index, query in enumerate(negatives):
        positions[query.instruction_id].append(index)
    return positions


def _query_group_maps(
    groups: Sequence[GroupSpec],
) -> tuple[dict[str, GroupSpec], dict[str, GroupSpec]]:
    by_id = {group.group_id: group for group in groups}
    by_instruction: dict[str, GroupSpec] = {}
    for group in groups:
        for instruction_id in group.instruction_ids:
            if instruction_id in by_instruction:
                raise ValueError(
                    f"instruction occurs in two V7 groups: {instruction_id}"
                )
            by_instruction[instruction_id] = group
    return by_id, by_instruction


def native_candidate_ids(
    artifact: ModelArtifacts,
    query_id: str,
) -> tuple[str, ...]:
    row = artifact.key_to_index[official_key(query_id)]
    return tuple(artifact.signatures[artifact.top_indices[row]].astype(str).tolist())


def build_shared_union_pool(
    first: ModelArtifacts,
    second: ModelArtifacts,
    query_ids: Iterable[str],
) -> tuple[dict[str, tuple[str, ...]], str]:
    if set(first.signatures.tolist()) != set(second.signatures.tolist()):
        raise ValueError(
            f"cannot share candidates: {first.spec.name} and {second.spec.name} "
            "do not have identical corpus identifiers"
        )
    pools: dict[str, tuple[str, ...]] = {}
    for query_id in query_ids:
        pools[query_id] = tuple(
            dict.fromkeys(
                (
                    *native_candidate_ids(first, query_id),
                    *native_candidate_ids(second, query_id),
                )
            )
        )
    fingerprint = stable_hash(
        {
            "models": [first.spec.name, second.spec.name],
            "queries": [
                [query_id, list(pools[query_id])] for query_id in sorted(pools)
            ],
        }
    )
    return pools, fingerprint


def build_candidate_profile_frame(
    artifact: ModelArtifacts,
    ground_truth: pd.DataFrame,
    groups: Sequence[GroupSpec],
    positive_queries: Sequence[QuerySpec],
    negatives: Sequence[QuerySpec],
    negative_embeddings: np.ndarray,
    coverage_frame: pd.DataFrame,
    validated_positive_ids: dict[str, set[str]],
    *,
    pool_name: str,
    candidate_pools: dict[str, tuple[str, ...]] | None,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build candidate-level profiles without requiring a complete edit grid.

    ``candidate_pools=None`` selects the model's native Top-K.  Official
    positives missing from that pool are appended only for supervised
    diagnostics; ``in_pool`` remains false, so reranking never receives them.
    """

    if len(negative_embeddings) != len(negatives):
        raise ValueError(
            f"negative embedding count mismatch: {len(negative_embeddings)} != "
            f"{len(negatives)}"
        )
    _, group_by_instruction = _query_group_maps(groups)
    negative_positions = _query_negative_positions(negatives)
    coverage_by_query = coverage_frame.set_index("query_id").to_dict("index")
    missing_official_queries: list[str] = []
    missing_positive_images: Counter[str] = Counter()
    rows: list[pd.DataFrame] = []

    for query in positive_queries:
        query_id = normalize_query_id(query.instruction_id)
        if query_id not in ground_truth.index:
            missing_official_queries.append(query_id)
            continue
        key = official_key(query_id)
        if key not in artifact.key_to_index:
            raise KeyError(f"{artifact.spec.name} lacks official query embedding {key}")
        query_row = artifact.key_to_index[key]
        gt_row = ground_truth.loc[query_id]
        group = group_by_instruction[query.instruction_id]

        native_ids = tuple(
            artifact.signatures[artifact.top_indices[query_row]].astype(str).tolist()
        )
        native_rank = {value: rank for rank, value in enumerate(native_ids, start=1)}
        if candidate_pools is None:
            pool_ids = native_ids
        else:
            if query_id not in candidate_pools:
                raise KeyError(f"candidate pool lacks pilot query {query_id}")
            pool_ids = tuple(dict.fromkeys(candidate_pools[query_id]))

        positive_ids = clean_identifier_values(gt_row["positive_candidates"])
        negative_ids = clean_identifier_values(gt_row["negative_candidates"])
        positive_set = set(positive_ids)
        # Positive precedence makes the diagnostic classes mutually exclusive.
        official_negative_set = set(negative_ids) - positive_set
        usable_pool: list[str] = []
        for candidate_id in pool_ids:
            if candidate_id not in artifact.signature_to_index:
                raise KeyError(
                    f"{artifact.spec.name} shared pool candidate is absent: {candidate_id}"
                )
            usable_pool.append(candidate_id)
        candidate_ids = list(usable_pool)
        for candidate_id in positive_ids:
            if candidate_id not in artifact.signature_to_index:
                missing_positive_images[query_id] += 1
            elif candidate_id not in candidate_ids:
                candidate_ids.append(candidate_id)
        if not candidate_ids:
            continue

        candidate_indices = np.asarray(
            [artifact.signature_to_index[value] for value in candidate_ids],
            dtype=np.int64,
        )
        candidate_embeddings = np.asarray(
            artifact.corpus[candidate_indices], dtype=np.float32
        )
        positive_query = np.asarray(
            artifact.positive_queries[query_row], dtype=np.float32
        )
        positions = negative_positions.get(query.instruction_id, [])
        query_negatives = np.asarray(negative_embeddings[positions], dtype=np.float32)
        query_negative_categories = [
            response_category(negatives[position]) for position in positions
        ]
        response = summarize_candidate_responses(
            positive_query,
            candidate_embeddings,
            query_negatives,
            query_negative_categories,
        )

        # A category-assignment placebo.  The edit-level scores are held fixed;
        # only their category grouping is shuffled deterministically per query.
        permuted_categories = list(query_negative_categories)
        permutation_seed = int(
            stable_hash({"query_id": query_id, "seed": seed})[:16], 16
        )
        np.random.default_rng(permutation_seed).shuffle(permuted_categories)
        permuted_response = summarize_candidate_responses(
            positive_query,
            candidate_embeddings,
            query_negatives,
            permuted_categories,
        )

        group_query_rows = [
            artifact.key_to_index[official_key(normalize_query_id(instruction_id))]
            for instruction_id in group.instruction_ids
        ]
        group_positive_queries = np.asarray(
            artifact.positive_queries[group_query_rows], dtype=np.float32
        )
        source_position = group.instruction_ids.index(query.instruction_id)
        paraphrase = summarize_positive_paraphrases(
            positive_query,
            group_positive_queries,
            candidate_embeddings,
            source_position=source_position,
        )

        available_positive_indices = [
            artifact.signature_to_index[value]
            for value in positive_ids
            if value in artifact.signature_to_index
        ]
        if available_positive_indices:
            positive_scores = (
                np.asarray(
                    artifact.corpus[available_positive_indices], dtype=np.float32
                )
                @ positive_query
            )
            best_positive_score = float(np.max(positive_scores))
            worst_positive_score = float(np.min(positive_scores))
        else:
            best_positive_score = math.nan
            worst_positive_score = math.nan

        pool_rank = {value: rank for rank, value in enumerate(usable_pool, start=1)}
        validated = validated_positive_ids.get(group.group_id, set())
        is_positive = np.asarray(
            [value in positive_set for value in candidate_ids], dtype=bool
        )
        is_official_negative = np.asarray(
            [value in official_negative_set for value in candidate_ids], dtype=bool
        )
        is_unlabeled = ~(is_positive | is_official_negative)
        sim_positive = response["sim_positive"]
        per_query = pd.DataFrame(
            {
                "model": artifact.spec.name,
                "pool_name": pool_name,
                "group_id": group.group_id,
                "query_id": query_id,
                "instruction_id": query.instruction_id,
                "candidate_id": candidate_ids,
                "candidate_index": candidate_indices,
                "pool_rank": [pool_rank.get(value, np.nan) for value in candidate_ids],
                "native_rank": [
                    native_rank.get(value, np.nan) for value in candidate_ids
                ],
                "in_pool": [value in pool_rank for value in candidate_ids],
                "in_native_top10": [
                    native_rank.get(value, 10**9) <= 10 for value in candidate_ids
                ],
                "in_native_top20": [
                    native_rank.get(value, 10**9) <= 20 for value in candidate_ids
                ],
                "in_native_top50": [
                    native_rank.get(value, 10**9) <= 50 for value in candidate_ids
                ],
                "candidate_source": [
                    "retrieved_pool" if value in pool_rank else "appended_positive"
                    for value in candidate_ids
                ],
                "is_positive": is_positive,
                "is_official_negative": is_official_negative,
                "is_unlabeled": is_unlabeled,
                "label_clean": np.where(
                    is_positive, 1, np.where(is_official_negative, 0, -1)
                ),
                "positive_validation_status": [
                    "validated"
                    if value in positive_set and value in validated
                    else "unseen_positive"
                    if value in positive_set
                    else "not_positive"
                    for value in candidate_ids
                ],
                "best_positive_score": best_positive_score,
                "worst_positive_score": worst_positive_score,
                "official_negative_above_best_positive": (
                    is_official_negative & (sim_positive > best_positive_score)
                ),
                "official_negative_above_any_positive": (
                    is_official_negative & (sim_positive > worst_positive_score)
                ),
                "unlabeled_competitor_above_best_positive": (
                    is_unlabeled & (sim_positive > best_positive_score)
                ),
                "official_positive_count": len(positive_ids),
                "official_negative_count_clean": len(official_negative_set),
                "official_negative_count_raw": len(
                    gt_row["negative_candidates"].tolist()
                    if hasattr(gt_row["negative_candidates"], "tolist")
                    else gt_row["negative_candidates"]
                ),
            }
        )
        for column, values in response.items():
            per_query[column] = values
        for column, values in paraphrase.items():
            per_query[column] = values
        per_query["sim_other_positive_mean"] = (
            per_query["sim_positive"] - per_query["paraphrase_delta_mean"]
        )
        per_query["d_group_available"] = (
            per_query["d_available"]
            + per_query["sim_positive_ensemble"]
            - per_query["sim_positive"]
        )
        per_query["abs_paraphrase_delta_mean"] = per_query[
            "paraphrase_delta_mean"
        ].abs()
        per_query["d_available_minus_abs_paraphrase"] = (
            per_query["d_available"] - per_query["abs_paraphrase_delta_mean"]
        )
        per_query["d_category_macro_weighting_permuted"] = permuted_response[
            "d_available"
        ]
        for source_column, target_column in zip(
            RESPONSE_COLUMNS, PERMUTED_RESPONSE_COLUMNS, strict=True
        ):
            per_query[target_column] = permuted_response[source_column]
        for category, mask_column in zip(
            RESPONSE_CATEGORIES, PERMUTED_MASK_COLUMNS, strict=True
        ):
            per_query[mask_column] = category in permuted_categories

        coverage = coverage_by_query[query.instruction_id]
        for category, mask_column in zip(
            RESPONSE_CATEGORIES, MASK_COLUMNS, strict=True
        ):
            per_query[mask_column] = bool(coverage[f"has_{category}"])
            per_query[f"negative_count_{category}"] = int(coverage[f"count_{category}"])
        per_query["strict_edit_count"] = int(coverage["strict_edit_count"])
        per_query["available_category_count"] = int(
            coverage["available_category_count"]
        )
        per_query["has_any_strict"] = bool(coverage["has_any_strict"])
        per_query["has_object_pair"] = bool(coverage["has_object_pair"])
        per_query["has_complete_6d"] = bool(coverage["has_complete_6d"])
        rows.append(per_query)

    if missing_official_queries:
        raise KeyError(
            f"V7 queries missing from official ground truth: {missing_official_queries[:5]}"
        )
    if not rows:
        raise ValueError(f"no candidate profiles were built for {artifact.spec.name}")
    frame = pd.concat(rows, ignore_index=True)
    frame["d_available_shuffled"] = candidate_shuffle_within_query(
        frame, "d_available", seed=seed, strata=("in_pool",)
    )
    frame.insert(0, "row_id", np.arange(len(frame), dtype=np.int64))
    summary = {
        "model": artifact.spec.name,
        "pool_name": pool_name,
        "candidate_row_count": len(frame),
        "query_count": int(frame["query_id"].nunique()),
        "group_count": int(frame["group_id"].nunique()),
        "retrieved_pool_row_count": int(frame["in_pool"].sum()),
        "appended_positive_row_count": int((~frame["in_pool"]).sum()),
        "missing_official_positive_image_count": int(
            sum(missing_positive_images.values())
        ),
        "query_with_missing_positive_image_count": len(missing_positive_images),
    }
    return frame, summary


def negative_text_sensitivity_tables(
    artifact: ModelArtifacts,
    ground_truth: pd.DataFrame,
    negatives: Sequence[QuerySpec],
    negative_embeddings: np.ndarray,
    *,
    bootstrap_replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute the existing target Delta/win-rate baseline from the same q- cache."""

    rows = []
    for query, negative_embedding in zip(negatives, negative_embeddings, strict=True):
        query_id = normalize_query_id(query.instruction_id)
        gt_row = ground_truth.loc[query_id]
        positive_ids = [
            value
            for value in clean_identifier_values(gt_row["positive_candidates"])
            if value in artifact.signature_to_index
        ]
        if not positive_ids:
            continue
        target_indices = [artifact.signature_to_index[value] for value in positive_ids]
        targets = np.asarray(artifact.corpus[target_indices], dtype=np.float32)
        positive_query = np.asarray(
            artifact.positive_queries[artifact.key_to_index[official_key(query_id)]],
            dtype=np.float32,
        )
        positive_scores = targets @ positive_query
        negative_scores = targets @ np.asarray(negative_embedding, dtype=np.float32)
        deltas = positive_scores - negative_scores
        rows.append(
            {
                "model": artifact.spec.name,
                "group_id": query.group_id,
                "query_id": query_id,
                "instruction_id": query.instruction_id,
                "negative_id": query.negative_id,
                "semantic_target": query.semantic_target,
                "edit_operation": query.edit_operation,
                "response_category": response_category(query),
                "positive_target_count": len(positive_ids),
                "mean_positive_query_target_score": float(positive_scores.mean()),
                "mean_negative_query_target_score": float(negative_scores.mean()),
                "mean_target_delta": float(deltas.mean()),
                "target_raw_win_rate": float(np.mean(deltas > 0)),
            }
        )
    pair_frame = pd.DataFrame(rows)
    subset_masks: dict[str, pd.Series] = {
        "strict": pd.Series(True, index=pair_frame.index),
        "add": pair_frame["edit_operation"].eq("add"),
        "substitute": pair_frame["edit_operation"].eq("substitute"),
    }
    subset_masks.update(
        {
            category: pair_frame["response_category"].eq(category)
            for category in RESPONSE_CATEGORIES
        }
    )
    summary_rows = []
    for subset, mask in subset_masks.items():
        selected = pair_frame.loc[mask]
        if selected.empty:
            continue
        instruction_values = (
            selected.groupby(["group_id", "instruction_id"], sort=False)[
                ["mean_target_delta", "target_raw_win_rate"]
            ]
            .mean()
            .reset_index()
            .rename(columns={"instruction_id": "query_id"})
        )
        delta_summary = group_macro_bootstrap(
            instruction_values,
            value_column="mean_target_delta",
            replicates=bootstrap_replicates,
            seed=seed + int(stable_hash([artifact.spec.name, subset, "delta"])[:8], 16),
        )
        win_summary = group_macro_bootstrap(
            instruction_values,
            value_column="target_raw_win_rate",
            replicates=bootstrap_replicates,
            seed=seed + int(stable_hash([artifact.spec.name, subset, "win"])[:8], 16),
        )
        summary_rows.append(
            {
                "model": artifact.spec.name,
                "subset": subset,
                "group_count": int(selected["group_id"].nunique()),
                "instruction_count": int(selected["instruction_id"].nunique()),
                "negative_count": len(selected),
                "positive_target_comparison_count": int(
                    selected["positive_target_count"].sum()
                ),
                "pair_balanced_mean_target_delta": delta_summary["estimate"],
                "mean_target_delta_ci_low": delta_summary["ci_low"],
                "mean_target_delta_ci_high": delta_summary["ci_high"],
                "pair_balanced_target_raw_win_rate": win_summary["estimate"],
                "target_raw_win_rate_ci_low": win_summary["ci_low"],
                "target_raw_win_rate_ci_high": win_summary["ci_high"],
            }
        )
    return pair_frame, pd.DataFrame(summary_rows)


def group_macro_bootstrap(
    query_values: pd.DataFrame,
    *,
    value_column: str,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    numeric_values = pd.to_numeric(query_values[value_column], errors="coerce")
    usable = query_values.loc[
        np.isfinite(numeric_values), ["group_id", "query_id", value_column]
    ].copy()
    usable[value_column] = numeric_values.loc[usable.index].astype(float)
    group_values = usable.groupby("group_id", sort=False)[value_column].mean()
    if group_values.empty:
        return {
            "estimate": math.nan,
            "ci_low": math.nan,
            "ci_high": math.nan,
            "random_sign_p_two_sided": math.nan,
            "query_count": 0,
            "group_count": 0,
        }
    values = group_values.to_numpy(dtype=np.float64)
    estimate = float(np.mean(values))
    if replicates <= 0:
        low = high = p_value = math.nan
    else:
        rng = np.random.default_rng(seed)
        sampled = rng.integers(0, len(values), size=(replicates, len(values)))
        bootstrap_values = values[sampled].mean(axis=1)
        low, high = np.quantile(bootstrap_values, [0.025, 0.975]).tolist()
        random_signs = rng.choice(
            np.asarray([-1.0, 1.0]),
            size=(replicates, len(values)),
        )
        null_values = (random_signs * values[None, :]).mean(axis=1)
        p_value = float(
            (np.count_nonzero(np.abs(null_values) >= abs(estimate)) + 1)
            / (replicates + 1)
        )
    return {
        "estimate": estimate,
        "ci_low": float(low),
        "ci_high": float(high),
        "random_sign_p_two_sided": float(p_value),
        "query_count": int(usable["query_id"].nunique()),
        "group_count": int(len(group_values)),
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    output = np.full(len(values), np.nan, dtype=np.float64)
    finite_positions = np.flatnonzero(np.isfinite(values))
    if not len(finite_positions):
        return output.tolist()
    ordered = finite_positions[np.argsort(values[finite_positions])]
    running = 0.0
    count = len(ordered)
    for order, position in enumerate(ordered):
        adjusted = min(1.0, (count - order) * values[position])
        running = max(running, adjusted)
        output[position] = running
    return output.tolist()


def _within_query_difference(
    profile: pd.DataFrame,
    dimension: str,
    first_mask: pd.Series,
    second_mask: pd.Series,
) -> tuple[pd.DataFrame, int, int]:
    finite = np.isfinite(profile[dimension])
    first = profile.loc[first_mask & finite]
    second = profile.loc[second_mask & finite]
    first_means = first.groupby(["group_id", "query_id"], sort=False)[dimension].mean()
    second_means = second.groupby(["group_id", "query_id"], sort=False)[
        dimension
    ].mean()
    joined = pd.concat(
        {"first": first_means, "second": second_means}, axis=1, join="inner"
    ).dropna()
    if joined.empty:
        return (
            pd.DataFrame(
                columns=["group_id", "query_id", "first", "second", "difference"]
            ),
            0,
            0,
        )
    joined["difference"] = joined["first"] - joined["second"]
    return (
        joined.reset_index()[["group_id", "query_id", "first", "second", "difference"]],
        int(
            first.loc[
                first.set_index(["group_id", "query_id"]).index.isin(joined.index)
            ].shape[0]
        ),
        int(
            second.loc[
                second.set_index(["group_id", "query_id"]).index.isin(joined.index)
            ].shape[0]
        ),
    )


def candidate_dimension_statistics(
    profile: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    contrasts = {
        "retrieved_positive_minus_official_negative": (
            profile["in_pool"] & profile["is_positive"],
            profile["in_pool"] & profile["is_official_negative"],
        ),
        "all_positive_minus_retrieved_official_negative": (
            profile["is_positive"],
            profile["in_pool"] & profile["is_official_negative"],
        ),
        "all_positive_minus_retrieved_unlabeled": (
            profile["is_positive"],
            profile["in_pool"] & profile["is_unlabeled"],
        ),
        "all_positive_minus_retrieved_nonpositive_benchmark": (
            profile["is_positive"],
            profile["in_pool"] & ~profile["is_positive"],
        ),
        "retrieved_positive_minus_retrieved_nonpositive_benchmark": (
            profile["in_pool"] & profile["is_positive"],
            profile["in_pool"] & ~profile["is_positive"],
        ),
        "retrieved_positive_minus_appended_positive": (
            profile["in_pool"] & profile["is_positive"],
            ~profile["in_pool"] & profile["is_positive"],
        ),
        "all_positive_minus_rank_violating_official_negative": (
            profile["is_positive"],
            profile["in_pool"] & profile["official_negative_above_best_positive"],
        ),
        "validated_minus_unseen_positive": (
            profile["positive_validation_status"].eq("validated"),
            profile["positive_validation_status"].eq("unseen_positive"),
        ),
    }
    dimensions = (
        *PROFILE_DIMENSIONS,
        "d_group_available",
        "d_category_macro_weighting_permuted",
        "d_available_shuffled",
        "paraphrase_delta_mean",
    )
    rows: list[dict[str, Any]] = []
    query_values: dict[str, pd.DataFrame] = {}
    for contrast, (first_mask, second_mask) in contrasts.items():
        contrast_start = len(rows)
        for dimension in dimensions:
            differences, first_count, second_count = _within_query_difference(
                profile,
                dimension,
                first_mask,
                second_mask,
            )
            stat_seed = seed + int(stable_hash([contrast, dimension])[:8], 16)
            summary = group_macro_bootstrap(
                differences,
                value_column="difference",
                replicates=bootstrap_replicates,
                seed=stat_seed,
            )
            first_summary = group_macro_bootstrap(
                differences,
                value_column="first",
                replicates=bootstrap_replicates,
                seed=stat_seed + 1,
            )
            second_summary = group_macro_bootstrap(
                differences,
                value_column="second",
                replicates=bootstrap_replicates,
                seed=stat_seed + 2,
            )
            rows.append(
                {
                    "model": profile["model"].iloc[0],
                    "pool_name": profile["pool_name"].iloc[0],
                    "contrast": contrast,
                    "dimension": dimension,
                    "first_candidate_count": first_count,
                    "second_candidate_count": second_count,
                    "first_mean": first_summary["estimate"],
                    "first_mean_ci_low": first_summary["ci_low"],
                    "first_mean_ci_high": first_summary["ci_high"],
                    "second_mean": second_summary["estimate"],
                    "second_mean_ci_low": second_summary["ci_low"],
                    "second_mean_ci_high": second_summary["ci_high"],
                    **summary,
                }
            )
            query_values[f"{contrast}::{dimension}"] = differences
        p_values = [row["random_sign_p_two_sided"] for row in rows[contrast_start:]]
        for row, adjusted in zip(
            rows[contrast_start:], holm_adjust(p_values), strict=True
        ):
            row["holm_p"] = adjusted
    return pd.DataFrame(rows), query_values


def rank_violation_frame(profile: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (group_id, query_id), query_frame in profile.groupby(
        ["group_id", "query_id"], sort=False
    ):
        positives = query_frame.loc[
            query_frame["is_positive"], "sim_positive"
        ].to_numpy()
        retrieved = query_frame.loc[query_frame["in_pool"]]
        negatives = retrieved.loc[
            retrieved["is_official_negative"], "sim_positive"
        ].to_numpy()
        unlabeled = retrieved.loc[retrieved["is_unlabeled"], "sim_positive"].to_numpy()
        if len(positives) and len(negatives):
            comparisons = negatives[:, None] - positives[None, :]
            pairwise_inversion = float(
                np.mean((comparisons > 0) + 0.5 * (comparisons == 0))
            )
            negative_above_best = int(np.count_nonzero(negatives > positives.max()))
            negative_above_any = int(np.count_nonzero(negatives > positives.min()))
        else:
            pairwise_inversion = math.nan
            negative_above_best = 0
            negative_above_any = 0
        unlabeled_above_best = (
            int(np.count_nonzero(unlabeled > positives.max())) if len(positives) else 0
        )
        rows.append(
            {
                "model": query_frame["model"].iloc[0],
                "pool_name": query_frame["pool_name"].iloc[0],
                "group_id": group_id,
                "query_id": query_id,
                "positive_count_available": len(positives),
                "retrieved_positive_count": int((retrieved["is_positive"]).sum()),
                "retrieved_official_negative_count": len(negatives),
                "retrieved_unlabeled_count": len(unlabeled),
                "official_negative_positive_pair_count": len(positives)
                * len(negatives),
                "pairwise_official_negative_inversion_rate": pairwise_inversion,
                "official_negative_above_best_positive_count": negative_above_best,
                "official_negative_above_any_positive_count": negative_above_any,
                "unlabeled_competitor_above_best_positive_count": unlabeled_above_best,
                "has_positive_in_native_top10": bool(
                    (query_frame["is_positive"] & query_frame["in_native_top10"]).any()
                ),
                "has_official_negative_in_native_top10": bool(
                    (
                        query_frame["is_official_negative"]
                        & query_frame["in_native_top10"]
                    ).any()
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_rank_violations(
    frame: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    seed: int,
) -> pd.DataFrame:
    columns = (
        "pairwise_official_negative_inversion_rate",
        "official_negative_above_best_positive_count",
        "official_negative_above_any_positive_count",
        "unlabeled_competitor_above_best_positive_count",
        "has_positive_in_native_top10",
        "has_official_negative_in_native_top10",
    )
    rows = []
    for column in columns:
        values = frame[["group_id", "query_id", column]].copy()
        values[column] = values[column].astype(float)
        summary = group_macro_bootstrap(
            values,
            value_column=column,
            replicates=bootstrap_replicates,
            seed=seed + int(stable_hash(column)[:8], 16),
        )
        rows.append(
            {
                "model": frame["model"].iloc[0],
                "pool_name": frame["pool_name"].iloc[0],
                "metric": column,
                **summary,
            }
        )
    return pd.DataFrame(rows)


def default_feature_specs() -> tuple[FeatureSpec, ...]:
    return (
        FeatureSpec("source_similarity", "all", ("sim_positive",)),
        FeatureSpec(
            "positive_ensemble",
            "all",
            ("sim_positive_ensemble",),
        ),
        FeatureSpec(
            "source_plus_paraphrase",
            "all",
            ("sim_positive", "paraphrase_delta_mean"),
        ),
        FeatureSpec(
            "source_similarity_strict",
            "strict_any",
            ("sim_positive",),
        ),
        FeatureSpec(
            "positive_ensemble_strict",
            "strict_any",
            ("sim_positive_ensemble",),
        ),
        FeatureSpec(
            "source_plus_paraphrase_strict",
            "strict_any",
            ("sim_positive", "paraphrase_delta_mean"),
        ),
        FeatureSpec("d_available_only", "strict_any", ("d_available",)),
        FeatureSpec(
            "source_plus_d_available",
            "strict_any",
            ("sim_positive", "d_available"),
        ),
        FeatureSpec(
            "ensemble_plus_d_group",
            "strict_any",
            ("sim_positive_ensemble", "d_group_available"),
        ),
        FeatureSpec(
            "source_paraphrase_and_d",
            "strict_any",
            ("sim_positive", "paraphrase_delta_mean", "d_available"),
        ),
        FeatureSpec(
            "response_moments",
            "strict_any",
            ("sim_positive", "d_available", "d_min", "d_max", "d_std"),
        ),
        FeatureSpec("mask_only", "strict_any", MASK_COLUMNS),
        FeatureSpec(
            "imputed_6d_values_only",
            "strict_any",
            RESPONSE_COLUMNS,
            impute=True,
        ),
        FeatureSpec(
            "masked_6d",
            "strict_any",
            (*RESPONSE_COLUMNS, *MASK_COLUMNS),
            impute=True,
        ),
        FeatureSpec(
            "source_plus_masked_6d",
            "strict_any",
            ("sim_positive", *RESPONSE_COLUMNS, *MASK_COLUMNS),
            impute=True,
        ),
        FeatureSpec(
            "category_permuted_6d",
            "strict_any",
            (*PERMUTED_RESPONSE_COLUMNS, *PERMUTED_MASK_COLUMNS),
            impute=True,
        ),
        FeatureSpec(
            "source_plus_category_permuted_6d",
            "strict_any",
            ("sim_positive", *PERMUTED_RESPONSE_COLUMNS, *PERMUTED_MASK_COLUMNS),
            impute=True,
        ),
        FeatureSpec(
            "candidate_shuffled_placebo",
            "strict_any",
            ("sim_positive", "d_available_shuffled"),
        ),
        FeatureSpec(
            "source_similarity_object_pair",
            "object_pair",
            ("sim_positive",),
        ),
        FeatureSpec(
            "positive_ensemble_object_pair",
            "object_pair",
            ("sim_positive_ensemble",),
        ),
        FeatureSpec(
            "object_pair_2d",
            "object_pair",
            ("d_object_add", "d_object_substitute"),
        ),
        FeatureSpec(
            "source_plus_object_pair_2d",
            "object_pair",
            ("sim_positive", "d_object_add", "d_object_substitute"),
        ),
    )


def _feature_cohort_mask(profile: pd.DataFrame, cohort: str) -> pd.Series:
    if cohort == "all":
        return pd.Series(True, index=profile.index)
    if cohort == "strict_any":
        return profile["has_any_strict"].astype(bool)
    if cohort == "object_pair":
        return profile["has_object_pair"].astype(bool)
    raise ValueError(f"unknown feature cohort: {cohort}")


def _feature_usable_mask(
    profile: pd.DataFrame,
    feature: FeatureSpec,
) -> pd.Series:
    mask = _feature_cohort_mask(profile, feature.cohort)
    if not feature.impute:
        mask &= np.isfinite(
            profile.loc[:, list(feature.columns)].to_numpy(dtype=np.float64)
        ).all(axis=1)
    return mask


def _scope_clean_mask(profile: pd.DataFrame, scope: str) -> pd.Series:
    if scope == "native_clean":
        return profile["in_pool"] & (
            profile["is_positive"] | profile["is_official_negative"]
        )
    if scope == "augmented_clean":
        return profile["is_positive"] | (
            profile["in_pool"] & profile["is_official_negative"]
        )
    if scope == "pool_nonpositive_benchmark":
        return profile["in_pool"]
    if scope == "rank_violating_official_negative":
        return profile["is_positive"] | (
            profile["in_pool"] & profile["official_negative_above_best_positive"]
        )
    if scope == "rank_violating_pool_nonpositive":
        return profile["is_positive"] | (
            profile["in_pool"]
            & ~profile["is_positive"]
            & (profile["sim_positive"] > profile["best_positive_score"])
        )
    raise ValueError(f"unknown probe scope: {scope}")


def _retain_queries_with_both_classes(frame: pd.DataFrame) -> pd.DataFrame:
    class_counts = frame.groupby("query_id", sort=False)["label_clean"].nunique()
    valid = set(class_counts[class_counts == 2].index)
    return frame.loc[frame["query_id"].isin(valid)].copy()


def query_balanced_class_weights(frame: pd.DataFrame) -> np.ndarray:
    weights = np.zeros(len(frame), dtype=np.float64)
    for _, group_frame in frame.groupby("group_id", sort=False):
        query_count = group_frame["query_id"].nunique()
        for _, query_frame in group_frame.groupby("query_id", sort=False):
            positions = frame.index.get_indexer(query_frame.index)
            labels = query_frame["label_clean"].to_numpy(dtype=np.int64)
            for label in (0, 1):
                label_positions = positions[labels == label]
                if len(label_positions):
                    weights[label_positions] = 0.5 / (
                        query_count * len(label_positions)
                    )
    if np.any(weights <= 0):
        raise ValueError("query-balanced weighting encountered a missing class")
    return weights


def _make_probe_pipeline(feature: FeatureSpec, *, seed: int) -> Pipeline:
    steps: list[tuple[str, Any]] = []
    if feature.impute:
        steps.append(
            (
                "imputer",
                SimpleImputer(strategy="median", keep_empty_features=True),
            )
        )
    steps.extend(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(C=1.0, max_iter=2000, random_state=seed),
            ),
        ]
    )
    return Pipeline(steps)


def _query_prediction_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (group_id, query_id), frame in predictions.groupby(
        ["group_id", "query_id"], sort=False
    ):
        labels = frame["label_clean"].to_numpy(dtype=np.int64)
        scores = frame["oof_probability"].to_numpy(dtype=np.float64)
        if len(np.unique(labels)) != 2:
            continue
        positive_scores = scores[labels == 1]
        negative_scores = scores[labels == 0]
        pairwise = positive_scores[:, None] - negative_scores[None, :]
        candidate_cohort = sorted(
            (
                str(candidate_id),
                int(label),
            )
            for candidate_id, label in zip(frame["candidate_id"], labels, strict=True)
        )
        rows.append(
            {
                "group_id": group_id,
                "query_id": query_id,
                "candidate_cohort_fingerprint": stable_hash(candidate_cohort),
                "query_ap": average_precision_score(labels, scores),
                "query_pairwise_auc": float(
                    np.mean((pairwise > 0) + 0.5 * (pairwise == 0))
                ),
            }
        )
    return pd.DataFrame(rows)


def run_grouped_oof_probe(
    profile: pd.DataFrame,
    feature: FeatureSpec,
    *,
    scope: str,
    fold_map: dict[str, int],
    bootstrap_replicates: int,
    seed: int,
    retain_pool_predictions: bool,
) -> ProbeOutput:
    usable_mask = _feature_usable_mask(profile, feature)
    clean = profile.loc[usable_mask & _scope_clean_mask(profile, scope)].copy()
    if scope in {
        "pool_nonpositive_benchmark",
        "rank_violating_pool_nonpositive",
    }:
        # This matches benchmark relevance (official positive vs every other
        # retrieved candidate), but nonpositive does not mean human-verified
        # semantic negative.  Keep the scope name explicit in every output.
        clean["label_clean"] = clean["is_positive"].astype(int)
    clean = _retain_queries_with_both_classes(clean).reset_index(drop=True)
    if clean.empty:
        raise ProbeNotEvaluableError(f"no two-class rows for {feature.name}/{scope}")
    unknown_groups = sorted(set(clean["group_id"]) - set(fold_map))
    if unknown_groups:
        raise KeyError(f"probe fold map lacks groups: {unknown_groups[:5]}")

    clean_predictions = np.full(len(clean), np.nan, dtype=np.float64)
    pool_prediction_parts: list[pd.DataFrame] = []
    fold_rows = []
    for fold in sorted(set(fold_map.values())):
        train_mask = clean["group_id"].map(fold_map).ne(fold).to_numpy()
        test_mask = ~train_mask
        train = clean.loc[train_mask]
        test = clean.loc[test_mask]
        if train.empty or test.empty or train["label_clean"].nunique() != 2:
            raise ProbeNotEvaluableError(
                f"invalid group fold {fold} for {feature.name}/{scope}: "
                f"train={len(train)}, test={len(test)}"
            )
        pipeline = _make_probe_pipeline(feature, seed=seed + fold)
        train_weights = query_balanced_class_weights(train.reset_index(drop=True))
        pipeline.fit(
            train.loc[:, list(feature.columns)],
            train["label_clean"].astype(int),
            classifier__sample_weight=train_weights,
        )
        clean_predictions[test_mask] = pipeline.predict_proba(
            test.loc[:, list(feature.columns)]
        )[:, 1]

        if retain_pool_predictions:
            pool_test_mask = (
                usable_mask
                & profile["in_pool"]
                & profile["group_id"].map(fold_map).eq(fold)
            )
            pool_test = profile.loc[pool_test_mask]
            prediction_part = pool_test[
                ["row_id", "group_id", "query_id", "candidate_id"]
            ].copy()
            prediction_part["oof_probability"] = pipeline.predict_proba(
                pool_test.loc[:, list(feature.columns)]
            )[:, 1]
            prediction_part["fold"] = fold
            pool_prediction_parts.append(prediction_part)
        fold_rows.append(
            {
                "fold": fold,
                "train_group_count": int(train["group_id"].nunique()),
                "test_group_count": int(test["group_id"].nunique()),
                "train_query_count": int(train["query_id"].nunique()),
                "test_query_count": int(test["query_id"].nunique()),
                "train_candidate_count": len(train),
                "test_candidate_count": len(test),
            }
        )
    if not np.isfinite(clean_predictions).all():
        raise RuntimeError(f"OOF predictions are incomplete for {feature.name}/{scope}")

    clean_output = clean[
        [
            "row_id",
            "group_id",
            "query_id",
            "candidate_id",
            "label_clean",
            "is_positive",
            "is_official_negative",
            "in_pool",
        ]
    ].copy()
    clean_output["oof_probability"] = clean_predictions
    clean_output["feature"] = feature.name
    clean_output["scope"] = scope
    metric_weights = query_balanced_class_weights(clean)
    labels = clean["label_clean"].to_numpy(dtype=np.int64)
    thresholded = (clean_predictions >= 0.5).astype(np.int64)
    query_metrics = _query_prediction_metrics(clean_output)
    ap_summary = group_macro_bootstrap(
        query_metrics,
        value_column="query_ap",
        replicates=bootstrap_replicates,
        seed=seed + int(stable_hash([feature.name, scope, "ap"])[:8], 16),
    )
    pairwise_summary = group_macro_bootstrap(
        query_metrics,
        value_column="query_pairwise_auc",
        replicates=bootstrap_replicates,
        seed=seed + int(stable_hash([feature.name, scope, "pairwise"])[:8], 16),
    )
    result = {
        "model": profile["model"].iloc[0],
        "pool_name": profile["pool_name"].iloc[0],
        "scope": scope,
        "feature": feature.name,
        "feature_columns": ",".join(feature.columns),
        "feature_cohort": feature.cohort,
        "uses_fold_local_imputation": feature.impute,
        "candidate_count": len(clean),
        "query_count": int(clean["query_id"].nunique()),
        "group_count": int(clean["group_id"].nunique()),
        "positive_candidate_count": int((labels == 1).sum()),
        "negative_candidate_count": int((labels == 0).sum()),
        "weighted_auroc": roc_auc_score(
            labels, clean_predictions, sample_weight=metric_weights
        ),
        "weighted_auprc": average_precision_score(
            labels, clean_predictions, sample_weight=metric_weights
        ),
        "weighted_accuracy_at_0_5": accuracy_score(
            labels, thresholded, sample_weight=metric_weights
        ),
        "weighted_balanced_accuracy_at_0_5": balanced_accuracy_score(
            labels, thresholded, sample_weight=metric_weights
        ),
        "query_macro_ap": ap_summary["estimate"],
        "query_macro_ap_ci_low": ap_summary["ci_low"],
        "query_macro_ap_ci_high": ap_summary["ci_high"],
        "query_macro_pairwise_auc": pairwise_summary["estimate"],
        "query_macro_pairwise_auc_ci_low": pairwise_summary["ci_low"],
        "query_macro_pairwise_auc_ci_high": pairwise_summary["ci_high"],
        "folds": fold_rows,
    }
    if pool_prediction_parts:
        pool_output = pd.concat(pool_prediction_parts, ignore_index=True)
        pool_output["feature"] = feature.name
        pool_output["scope"] = scope
    else:
        pool_output = pd.DataFrame()
    return ProbeOutput(result, clean_output, pool_output)


def probe_query_metrics_by_key(
    clean_predictions: pd.DataFrame,
) -> dict[tuple[str, str], pd.DataFrame]:
    output: dict[tuple[str, str], pd.DataFrame] = {}
    for (scope, feature), frame in clean_predictions.groupby(
        ["scope", "feature"], sort=False
    ):
        output[(str(scope), str(feature))] = _query_prediction_metrics(frame)
    return output


def paired_feature_probe_statistics(
    clean_predictions: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    seed: int,
) -> pd.DataFrame:
    query_metrics = probe_query_metrics_by_key(clean_predictions)
    rows = []
    model = (
        str(clean_predictions["model"].iloc[0]) if "model" in clean_predictions else ""
    )
    pool_name = (
        str(clean_predictions["pool_name"].iloc[0])
        if "pool_name" in clean_predictions
        else ""
    )
    for scope in (
        "native_clean",
        "pool_nonpositive_benchmark",
        "augmented_clean",
        *RANK_VIOLATION_PROBE_SCOPES,
    ):
        for enhanced_feature, control_feature in FEATURE_COMPARISONS:
            enhanced = query_metrics.get((scope, enhanced_feature))
            control = query_metrics.get((scope, control_feature))
            if enhanced is None or control is None:
                continue
            enhanced_ids = set(enhanced["query_id"])
            control_ids = set(control["query_id"])
            if enhanced_ids != control_ids:
                raise ValueError(
                    f"paired feature comparison changed cohort for {scope}: "
                    f"{enhanced_feature}={len(enhanced_ids)} vs "
                    f"{control_feature}={len(control_ids)}"
                )
            paired = enhanced.merge(
                control,
                on=["group_id", "query_id"],
                suffixes=("_enhanced", "_control"),
                validate="one_to_one",
            )
            if (
                not paired["candidate_cohort_fingerprint_enhanced"]
                .eq(paired["candidate_cohort_fingerprint_control"])
                .all()
            ):
                raise ValueError(
                    f"paired feature candidate/label cohort changed for {scope}: "
                    f"{enhanced_feature} vs {control_feature}"
                )
            for metric in ("query_ap", "query_pairwise_auc"):
                paired["delta"] = (
                    paired[f"{metric}_enhanced"] - paired[f"{metric}_control"]
                )
                summary = group_macro_bootstrap(
                    paired[["group_id", "query_id", "delta"]],
                    value_column="delta",
                    replicates=bootstrap_replicates,
                    seed=seed
                    + int(
                        stable_hash([scope, enhanced_feature, control_feature, metric])[
                            :8
                        ],
                        16,
                    ),
                )
                rows.append(
                    {
                        "model": model,
                        "pool_name": pool_name,
                        "scope": scope,
                        "enhanced_feature": enhanced_feature,
                        "control_feature": control_feature,
                        "metric": metric,
                        **summary,
                    }
                )
    return pd.DataFrame(rows)


def run_all_probes(
    profile: pd.DataFrame,
    *,
    fold_map: dict[str, int],
    bootstrap_replicates: int,
    seed: int,
    retain_primary_pool_predictions: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results: list[dict[str, Any]] = []
    clean_predictions: list[pd.DataFrame] = []
    primary_pool_predictions: list[pd.DataFrame] = []
    for scope in (
        "native_clean",
        "pool_nonpositive_benchmark",
        "augmented_clean",
        *RANK_VIOLATION_PROBE_SCOPES,
    ):
        for feature in default_feature_specs():
            if (
                scope in RANK_VIOLATION_PROBE_SCOPES
                and feature.name not in RANK_VIOLATION_FEATURES
            ):
                continue
            retain_pool = (
                retain_primary_pool_predictions
                and feature.name == PRIMARY_PROBE_FEATURE
                and scope in RERANK_PROBE_SCOPES
            )
            try:
                output = run_grouped_oof_probe(
                    profile,
                    feature,
                    scope=scope,
                    fold_map=fold_map,
                    bootstrap_replicates=bootstrap_replicates,
                    seed=seed,
                    retain_pool_predictions=retain_pool,
                )
            except ProbeNotEvaluableError as exc:
                results.append(
                    {
                        "model": profile["model"].iloc[0],
                        "pool_name": profile["pool_name"].iloc[0],
                        "scope": scope,
                        "feature": feature.name,
                        "feature_columns": ",".join(feature.columns),
                        "feature_cohort": feature.cohort,
                        "status": "not_evaluable",
                        "reason": str(exc),
                    }
                )
                continue
            result = dict(output.result)
            result.pop("folds")
            result["status"] = "ok"
            result["reason"] = ""
            results.append(result)
            clean_predictions.append(output.clean_predictions)
            if not output.pool_predictions.empty:
                primary_pool_predictions.append(output.pool_predictions)
    clean_frame = pd.concat(clean_predictions, ignore_index=True)
    # Attach identity columns after prediction to keep the per-row files
    # self-describing and enable strict paired-cohort checks.
    clean_frame["model"] = profile["model"].iloc[0]
    clean_frame["pool_name"] = profile["pool_name"].iloc[0]
    paired_frame = paired_feature_probe_statistics(
        clean_frame,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    return (
        pd.DataFrame(results),
        clean_frame,
        pd.concat(primary_pool_predictions, ignore_index=True)
        if primary_pool_predictions
        else pd.DataFrame(),
        paired_frame,
    )


def _raw_identifier_values(values: Any) -> list[Any]:
    if values is None:
        return []
    if hasattr(values, "tolist"):
        values = values.tolist()
    return list(values) if isinstance(values, list | tuple) else []


def official_ap_at_k(
    retrieved: Sequence[str], relevant: Sequence[Any], k: int
) -> float:
    if not relevant:
        return 0.0
    relevant_set = set(relevant)
    hits = 0
    total = 0.0
    for rank, candidate_id in enumerate(retrieved[:k], start=1):
        if candidate_id in relevant_set:
            hits += 1
            total += hits / rank
    return total / min(len(relevant), k)


def official_negative_recall_at_k(
    retrieved: Sequence[str],
    negatives: Sequence[Any],
    k: int,
) -> float:
    if not negatives:
        return 0.0
    negative_set = set(negatives)
    hits = sum(value in negative_set for value in retrieved[:k])
    return hits / min(len(negatives), k)


def positive_recall_at_k(
    retrieved: Sequence[str],
    positives: Sequence[Any],
    k: int,
) -> float:
    if not positives:
        return 0.0
    positive_set = set(positives)
    hits = sum(value in positive_set for value in retrieved[:k])
    return hits / min(len(positives), k)


def binary_ndcg_at_k(
    retrieved: Sequence[str],
    positives: Sequence[Any],
    k: int,
) -> float:
    if not positives:
        return 0.0
    positive_set = set(positives)
    gains = np.asarray(
        [1.0 if value in positive_set else 0.0 for value in retrieved[:k]],
        dtype=np.float64,
    )
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2, dtype=np.float64))
    dcg = float(np.sum(gains * discounts))
    ideal_hits = min(len(positive_set), k)
    ideal_dcg = float(np.sum(discounts[:ideal_hits]))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def ranking_set_jaccard(
    first: Sequence[Any],
    second: Sequence[Any],
    k: int,
) -> tuple[int, float, float, bool]:
    """Set overlap for two Top-K rankings; ordering is intentionally ignored."""

    if k <= 0 or len(first) < k or len(second) < k:
        raise ValueError("ranking_set_jaccard requires two rankings of length >= k")
    first_set = set(first[:k])
    second_set = set(second[:k])
    if len(first_set) != k or len(second_set) != k:
        raise ValueError("Top-K rankings contain duplicate candidates")
    overlap = len(first_set & second_set)
    union = len(first_set | second_set)
    return overlap, overlap / union, overlap / k, first_set == second_set


def per_query_ranking_jaccard_columns(
    reference_rankings: np.ndarray,
    variant_rankings: np.ndarray,
    *,
    ks: Sequence[int] = RANKING_JACCARD_KS,
) -> dict[str, np.ndarray]:
    if reference_rankings.shape != variant_rankings.shape:
        raise ValueError("ranking matrices must have identical shapes")
    output: dict[str, np.ndarray] = {}
    for k in ks:
        if k > reference_rankings.shape[1]:
            raise ValueError(f"Top-{k} exceeds ranking width")
        values = np.empty(len(reference_rankings), dtype=np.float64)
        for row in range(len(reference_rankings)):
            _, jaccard, _, _ = ranking_set_jaccard(
                reference_rankings[row], variant_rankings[row], k
            )
            values[row] = jaccard
        output[f"top{k}_jaccard_vs_native"] = values
    return output


def native_model_ranking_jaccard_tables(
    baseline: ModelArtifacts,
    hard: ModelArtifacts,
    ground_truth: pd.DataFrame,
    coverage_frame: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    seed: int,
    ks: Sequence[int] = RANKING_JACCARD_KS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare native candidate sets for two models on official query IDs."""

    query_to_group = {
        normalize_query_id(row.query_id): str(row.group_id)
        for row in coverage_frame.itertuples()
    }
    pilot_ids = set(query_to_group)
    strict_ids = {
        normalize_query_id(row.query_id)
        for row in coverage_frame.itertuples()
        if row.has_any_strict
    }
    object_pair_ids = {
        normalize_query_id(row.query_id)
        for row in coverage_frame.itertuples()
        if row.has_object_pair
    }
    rows: list[dict[str, Any]] = []
    for query_id in ground_truth.index:
        normalized_id = normalize_query_id(query_id)
        gt_row = ground_truth.loc[query_id]
        positives = _raw_identifier_values(gt_row["positive_candidates"])
        negatives = _raw_identifier_values(gt_row["negative_candidates"])
        positive_set = set(positives)
        baseline_row = baseline.key_to_index[official_key(normalized_id)]
        hard_row = hard.key_to_index[official_key(normalized_id)]
        baseline_ids = baseline.signatures[baseline.top_indices[baseline_row]].astype(
            str
        )
        hard_ids = hard.signatures[hard.top_indices[hard_row]].astype(str)
        for k in ks:
            overlap, jaccard, overlap_fraction, exact_match = ranking_set_jaccard(
                baseline_ids, hard_ids, k
            )
            rows.append(
                {
                    "baseline_model": baseline.spec.name,
                    "hard_model": hard.spec.name,
                    "query_id": normalized_id,
                    "group_id": query_to_group.get(normalized_id),
                    "k": k,
                    "overlap_count": overlap,
                    "overlap_fraction": overlap_fraction,
                    "jaccard": jaccard,
                    "exact_set_match": exact_match,
                    "baseline_positive_recall": positive_recall_at_k(
                        baseline_ids, positives, k
                    ),
                    "hard_positive_recall": positive_recall_at_k(
                        hard_ids, positives, k
                    ),
                    "baseline_positive_hit": float(
                        any(value in positive_set for value in baseline_ids[:k])
                    ),
                    "hard_positive_hit": float(
                        any(value in positive_set for value in hard_ids[:k])
                    ),
                    "baseline_negative_recall": official_negative_recall_at_k(
                        baseline_ids, negatives, k
                    ),
                    "hard_negative_recall": official_negative_recall_at_k(
                        hard_ids, negatives, k
                    ),
                    "baseline_ndcg": binary_ndcg_at_k(baseline_ids, positives, k),
                    "hard_ndcg": binary_ndcg_at_k(hard_ids, positives, k),
                }
            )
    per_query = pd.DataFrame(rows)
    for metric in (
        "positive_recall",
        "positive_hit",
        "negative_recall",
        "ndcg",
    ):
        per_query[f"delta_{metric}"] = (
            per_query[f"hard_{metric}"] - per_query[f"baseline_{metric}"]
        )
    cohorts = {
        "official_full_7635": set(ground_truth.index),
        "pilot_500_all_queries": pilot_ids,
        "pilot_strict_negative_eligible": strict_ids,
        "pilot_object_add_substitute_pair": object_pair_ids,
    }
    summary_rows: list[dict[str, Any]] = []
    for cohort, query_ids in cohorts.items():
        for k in ks:
            selected = per_query.loc[
                per_query["query_id"].isin(query_ids) & per_query["k"].eq(k)
            ].copy()
            bootstrap_unit = "query" if cohort == "official_full_7635" else "v7_group"
            bootstrap_groups = (
                selected["query_id"]
                if bootstrap_unit == "query"
                else selected["group_id"]
            )
            bootstrap_frame = selected[["query_id", "jaccard"]].copy()
            bootstrap_frame["group_id"] = bootstrap_groups.to_numpy()
            ci = group_macro_bootstrap(
                bootstrap_frame[["group_id", "query_id", "jaccard"]],
                value_column="jaccard",
                replicates=bootstrap_replicates,
                seed=seed
                + int(
                    stable_hash(
                        [baseline.spec.name, hard.spec.name, cohort, k, "jaccard"]
                    )[:8],
                    16,
                ),
            )
            summary_row: dict[str, Any] = {
                "baseline_model": baseline.spec.name,
                "hard_model": hard.spec.name,
                "cohort": cohort,
                "k": k,
                "query_count": len(selected),
                "bootstrap_unit": bootstrap_unit,
                "mean_jaccard": ci["estimate"],
                "query_mean_jaccard": float(selected["jaccard"].mean()),
                "jaccard_ci_low": ci["ci_low"],
                "jaccard_ci_high": ci["ci_high"],
                "median_jaccard": float(selected["jaccard"].median()),
                "mean_overlap_count": float(selected["overlap_count"].mean()),
                "mean_overlap_fraction": float(selected["overlap_fraction"].mean()),
                "exact_set_match_rate": float(selected["exact_set_match"].mean()),
            }
            for metric in (
                "positive_recall",
                "positive_hit",
                "negative_recall",
                "ndcg",
            ):
                delta_column = f"delta_{metric}"
                delta_frame = selected[["query_id", delta_column]].copy()
                delta_frame["group_id"] = bootstrap_groups.to_numpy()
                delta_ci = group_macro_bootstrap(
                    delta_frame[["group_id", "query_id", delta_column]],
                    value_column=delta_column,
                    replicates=bootstrap_replicates,
                    seed=seed
                    + int(
                        stable_hash(
                            [
                                baseline.spec.name,
                                hard.spec.name,
                                cohort,
                                k,
                                metric,
                            ]
                        )[:8],
                        16,
                    ),
                )
                summary_row[f"baseline_{metric}"] = float(
                    selected[f"baseline_{metric}"].mean()
                )
                summary_row[f"hard_{metric}"] = float(selected[f"hard_{metric}"].mean())
                summary_row[f"delta_{metric}"] = delta_ci["estimate"]
                summary_row[f"delta_{metric}_ci_low"] = delta_ci["ci_low"]
                summary_row[f"delta_{metric}_ci_high"] = delta_ci["ci_high"]
            summary_rows.append(summary_row)
    return per_query, pd.DataFrame(summary_rows)


def per_query_official_metrics(
    artifact: ModelArtifacts,
    rankings: np.ndarray,
    ground_truth: pd.DataFrame,
    *,
    k: int = 10,
) -> pd.DataFrame:
    if (
        rankings.ndim != 2
        or rankings.shape[0] != len(artifact.query_keys)
        or rankings.shape[1] < k
    ):
        raise ValueError("ranking rows do not match official query cache")
    if (
        rankings.min() < 0
        or rankings.max() >= len(artifact.signatures)
        or np.any(np.diff(np.sort(rankings, axis=1), axis=1) == 0)
    ):
        raise ValueError("rankings contain invalid or duplicate corpus indices")
    rows = []
    for query_id, gt_row in ground_truth.iterrows():
        key = official_key(query_id)
        if key not in artifact.key_to_index:
            raise KeyError(f"official query cache lacks {key}")
        query_row = artifact.key_to_index[key]
        retrieved = artifact.signatures[rankings[query_row, :k]].astype(str).tolist()
        positives = _raw_identifier_values(gt_row["positive_candidates"])
        negatives = _raw_identifier_values(gt_row["negative_candidates"])
        rows.append(
            {
                "query_id": query_id,
                "ap10": official_ap_at_k(retrieved, positives, k),
                "positive_recall10": positive_recall_at_k(retrieved, positives, k),
                "positive_hit10": float(
                    any(value in set(positives) for value in retrieved[:k])
                ),
                "ndcg10": binary_ndcg_at_k(retrieved, positives, k),
                "negative_recall10": official_negative_recall_at_k(
                    retrieved, negatives, k
                ),
            }
        )
    return pd.DataFrame(rows)


def cohort_metric_rows(
    metrics: pd.DataFrame,
    *,
    pilot_query_ids: set[str],
    strict_query_ids: set[str],
    object_pair_query_ids: set[str],
) -> list[dict[str, Any]]:
    cohorts = {
        "official_full_7635": pd.Series(True, index=metrics.index),
        "pilot_500_all_queries": metrics["query_id"].isin(pilot_query_ids),
        "pilot_strict_negative_eligible": metrics["query_id"].isin(strict_query_ids),
        "pilot_object_add_substitute_pair": metrics["query_id"].isin(
            object_pair_query_ids
        ),
    }
    rows = []
    for cohort, mask in cohorts.items():
        selected = metrics.loc[mask]
        row = {
            "cohort": cohort,
            "query_count": len(selected),
            "mAP@10": float(selected["ap10"].mean()),
            "PositiveRecall@10": float(selected["positive_recall10"].mean()),
            "PositiveHitRate@10": float(selected["positive_hit10"].mean()),
            "nDCG@10": float(selected["ndcg10"].mean()),
            "NegRecall@10": float(selected["negative_recall10"].mean()),
        }
        for k in RANKING_JACCARD_KS:
            column = f"top{k}_jaccard_vs_native"
            if column in selected:
                row[f"Top{k}JaccardVsNative"] = float(selected[column].mean())
        rows.append(row)
    return rows


def score_maps_from_predictions(
    predictions: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    if predictions.empty:
        return {}
    return {
        query_id: dict(
            zip(
                frame["candidate_id"],
                frame["oof_probability"].astype(float),
                strict=True,
            )
        )
        for query_id, frame in predictions.groupby("query_id", sort=False)
    }


def rerank_native_prefix(
    artifact: ModelArtifacts,
    score_maps: dict[str, dict[str, float]],
    *,
    depth: int,
) -> tuple[np.ndarray, set[str], set[str]]:
    if depth < 1 or depth > artifact.top_indices.shape[1]:
        raise ValueError(
            f"rerank depth must be in [1, {artifact.top_indices.shape[1]}], got {depth}"
        )
    rankings = np.asarray(artifact.top_indices).copy()
    eligible: set[str] = set()
    changed: set[str] = set()
    for query_id, scores_by_candidate in score_maps.items():
        key = official_key(query_id)
        if key not in artifact.key_to_index:
            continue
        row = artifact.key_to_index[key]
        prefix_indices = rankings[row, :depth].copy()
        prefix_ids = artifact.signatures[prefix_indices].astype(str).tolist()
        if any(value not in scores_by_candidate for value in prefix_ids):
            continue
        scores = np.asarray(
            [scores_by_candidate[value] for value in prefix_ids], dtype=np.float64
        )
        if not np.isfinite(scores).all():
            continue
        order = np.argsort(-scores, kind="stable")
        rankings[row, :depth] = prefix_indices[order]
        eligible.add(query_id)
        if not np.array_equal(order, np.arange(depth)):
            changed.add(query_id)
    return rankings, eligible, changed


def write_official_results(
    path: Path,
    artifact: ModelArtifacts,
    rankings: np.ndarray,
) -> None:
    results: dict[str, dict[str, list[str]]] = {}
    for row, key in enumerate(artifact.query_keys):
        key = str(key)
        if not key.startswith("official::"):
            raise ValueError(f"unexpected official query key: {key}")
        query_id = key.removeprefix("official::")
        results[query_id] = {
            "retrieved_items": artifact.signatures[rankings[row]].astype(str).tolist()
        }
    write_json_atomic(path, results)


def invoke_official_evaluator(
    *,
    pinpoint_root: Path,
    ground_truth: Path,
    results_path: Path,
    metrics_path: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(pinpoint_root / "evaluate.py"),
        "--results",
        str(results_path),
        "--ground_truth",
        str(ground_truth),
        "--output",
        str(metrics_path),
    ]
    print("[official-eval] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=pinpoint_root, check=True)
    frame = pd.read_csv(metrics_path)
    if len(frame) != 1:
        raise ValueError(f"expected one official evaluator row in {metrics_path}")
    return frame.iloc[0].to_dict()


def _safe_variant_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in value
    )


def _paired_group_delta_ci(
    baseline: pd.DataFrame,
    variant: pd.DataFrame,
    *,
    query_ids: set[str],
    query_to_group: dict[str, str],
    metric: str,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    columns = {
        "mAP@10": "ap10",
        "PositiveRecall@10": "positive_recall10",
        "PositiveHitRate@10": "positive_hit10",
        "nDCG@10": "ndcg10",
        "NegRecall@10": "negative_recall10",
    }
    if metric not in columns:
        raise ValueError(f"unsupported paired retrieval metric: {metric}")
    column = columns[metric]
    merged = baseline[["query_id", column]].merge(
        variant[["query_id", column]], on="query_id", suffixes=("_baseline", "_variant")
    )
    merged = merged.loc[merged["query_id"].isin(query_ids)].copy()
    merged["group_id"] = merged["query_id"].map(query_to_group)
    merged = merged.dropna(subset=["group_id"])
    merged["delta"] = merged[f"{column}_variant"] - merged[f"{column}_baseline"]
    return group_macro_bootstrap(
        merged[["group_id", "query_id", "delta"]],
        value_column="delta",
        replicates=replicates,
        seed=seed,
    )


def cross_model_positive_query_rankings(
    retriever: ModelArtifacts,
    reranker: ModelArtifacts,
    *,
    query_ids: Sequence[str],
    depth: int,
) -> tuple[np.ndarray, set[str]]:
    """Return retriever rankings reordered only by the other model's q+ score."""

    if depth <= 0 or depth > retriever.top_indices.shape[1]:
        raise ValueError("cross-model rerank depth exceeds retriever cache width")
    rankings = np.asarray(retriever.top_indices).copy()
    changed: set[str] = set()
    for query_id in query_ids:
        normalized_id = normalize_query_id(query_id)
        retriever_row = retriever.key_to_index[official_key(normalized_id)]
        reranker_row = reranker.key_to_index[official_key(normalized_id)]
        prefix = rankings[retriever_row, :depth].copy()
        candidate_ids = retriever.signatures[prefix].astype(str).tolist()
        try:
            reranker_candidate_indices = np.asarray(
                [reranker.signature_to_index[value] for value in candidate_ids],
                dtype=np.int64,
            )
        except KeyError as exc:
            raise KeyError(
                f"{reranker.spec.name} corpus lacks candidate from "
                f"{retriever.spec.name}: {exc}"
            ) from exc
        query_embedding = np.asarray(
            reranker.positive_queries[reranker_row], dtype=np.float32
        )
        candidate_embeddings = np.asarray(
            reranker.corpus[reranker_candidate_indices], dtype=np.float32
        )
        scores = candidate_embeddings @ query_embedding
        order = np.argsort(-scores, kind="stable")
        rankings[retriever_row, :depth] = prefix[order]
        if not np.array_equal(order, np.arange(depth)):
            changed.add(normalized_id)
    return rankings, changed


def cross_model_positive_query_rerank(
    retriever: ModelArtifacts,
    reranker: ModelArtifacts,
    ground_truth: pd.DataFrame,
    coverage_frame: pd.DataFrame,
    *,
    output_dir: Path,
    pinpoint_root: Path,
    ground_truth_path: Path,
    depth: int,
    bootstrap_replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rerank one model's native pool using another model's normal q+ score."""

    rankings, changed = cross_model_positive_query_rankings(
        retriever,
        reranker,
        query_ids=ground_truth.index.tolist(),
        depth=depth,
    )

    native_metrics = per_query_official_metrics(
        retriever, retriever.top_indices, ground_truth
    )
    reranked_metrics = per_query_official_metrics(retriever, rankings, ground_truth)
    query_rows = np.asarray(
        [
            retriever.key_to_index[official_key(query_id)]
            for query_id in reranked_metrics["query_id"]
        ],
        dtype=np.int64,
    )
    for column, values in per_query_ranking_jaccard_columns(
        np.asarray(retriever.top_indices), rankings
    ).items():
        reranked_metrics[column] = values[query_rows]

    query_to_group = {
        normalize_query_id(row.query_id): str(row.group_id)
        for row in coverage_frame.itertuples()
    }
    pilot_query_ids = set(query_to_group)
    strict_query_ids = {
        normalize_query_id(row.query_id)
        for row in coverage_frame.itertuples()
        if row.has_any_strict
    }
    object_pair_query_ids = {
        normalize_query_id(row.query_id)
        for row in coverage_frame.itertuples()
        if row.has_object_pair
    }
    native_cohorts = {
        row["cohort"]: row
        for row in cohort_metric_rows(
            native_metrics,
            pilot_query_ids=pilot_query_ids,
            strict_query_ids=strict_query_ids,
            object_pair_query_ids=object_pair_query_ids,
        )
    }
    cohort_ids = {
        "pilot_500_all_queries": pilot_query_ids,
        "pilot_strict_negative_eligible": strict_query_ids,
        "pilot_object_add_substitute_pair": object_pair_query_ids,
    }
    summary_rows: list[dict[str, Any]] = []
    for cohort_row in cohort_metric_rows(
        reranked_metrics,
        pilot_query_ids=pilot_query_ids,
        strict_query_ids=strict_query_ids,
        object_pair_query_ids=object_pair_query_ids,
    ):
        cohort = str(cohort_row["cohort"])
        native_row = native_cohorts[cohort]
        summary_row: dict[str, Any] = {
            **cohort_row,
            "retriever_model": retriever.spec.name,
            "reranker_model": reranker.spec.name,
            "variant": (
                f"{retriever.spec.name}__native_top{depth}__reranked_by__"
                f"{reranker.spec.name}__positive_query_similarity"
            ),
            "rerank_depth": depth,
            "rerank_changed_query_count": len(changed),
            "uses_negative_text": False,
            "uses_learned_probe": False,
        }
        for metric in (
            "mAP@10",
            "PositiveRecall@10",
            "PositiveHitRate@10",
            "nDCG@10",
            "NegRecall@10",
        ):
            summary_row[f"native_{metric}"] = native_row[metric]
            summary_row[f"delta_{metric}"] = cohort_row[metric] - native_row[metric]
            if cohort != "official_full_7635":
                ci = _paired_group_delta_ci(
                    native_metrics,
                    reranked_metrics,
                    query_ids=cohort_ids[cohort],
                    query_to_group=query_to_group,
                    metric=metric,
                    replicates=bootstrap_replicates,
                    seed=seed
                    + int(
                        stable_hash(
                            [
                                retriever.spec.name,
                                reranker.spec.name,
                                cohort,
                                metric,
                            ]
                        )[:8],
                        16,
                    ),
                )
                safe_metric = metric.replace("@", "_at_").replace("/", "_")
                summary_row[f"paired_delta_{safe_metric}"] = ci["estimate"]
                summary_row[f"paired_delta_{safe_metric}_ci_low"] = ci["ci_low"]
                summary_row[f"paired_delta_{safe_metric}_ci_high"] = ci["ci_high"]
        summary_rows.append(summary_row)

    variant_name = _safe_variant_name(
        f"{retriever.spec.name}__reranked_by__{reranker.spec.name}"
    )
    cross_dir = output_dir / "cross_model_rerank"
    cross_dir.mkdir(parents=True, exist_ok=True)
    results_path = cross_dir / f"{variant_name}.json"
    metrics_path = cross_dir / f"{variant_name}_official_metrics.csv"
    write_official_results(results_path, retriever, rankings)
    official_metrics = invoke_official_evaluator(
        pinpoint_root=pinpoint_root,
        ground_truth=ground_truth_path,
        results_path=results_path,
        metrics_path=metrics_path,
    )
    full_row = next(
        row for row in summary_rows if row["cohort"] == "official_full_7635"
    )
    if not math.isclose(
        float(official_metrics["mAP@10"]), float(full_row["mAP@10"]), abs_tol=5.1e-5
    ) or not math.isclose(
        float(official_metrics["NegRecall@10"]),
        float(full_row["NegRecall@10"]),
        abs_tol=5.1e-5,
    ):
        raise AssertionError("cross-model internal metrics disagree with evaluator")

    query_output = reranked_metrics.copy()
    query_output["retriever_model"] = retriever.spec.name
    query_output["reranker_model"] = reranker.spec.name
    query_output["group_id"] = query_output["query_id"].map(query_to_group)
    return pd.DataFrame(summary_rows), query_output


def run_rerank_experiments(
    artifact: ModelArtifacts,
    profile: pd.DataFrame,
    primary_pool_predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    coverage_frame: pd.DataFrame,
    *,
    output_dir: Path,
    pinpoint_root: Path,
    ground_truth_path: Path,
    rerank_depths: Sequence[int],
    bootstrap_replicates: int,
    seed: int,
    official_eval_all: bool,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, pd.DataFrame]]:
    model_dir = output_dir / artifact.spec.name / "rerank"
    model_dir.mkdir(parents=True, exist_ok=True)
    pilot_query_ids = set(profile["query_id"].unique())
    strict_query_ids = {
        normalize_query_id(row.query_id)
        for row in coverage_frame.itertuples()
        if row.has_any_strict
    }
    object_pair_query_ids = {
        normalize_query_id(row.query_id)
        for row in coverage_frame.itertuples()
        if row.has_object_pair
    }
    query_to_group = {
        normalize_query_id(row.query_id): str(row.group_id)
        for row in coverage_frame.itertuples()
    }
    baseline_rankings = np.asarray(artifact.top_indices).copy()
    baseline_metrics = per_query_official_metrics(
        artifact, baseline_rankings, ground_truth
    )

    variants: list[dict[str, Any]] = [
        {
            "variant": "native",
            "method": "native",
            "depth": None,
            "probe_scope": None,
            "rankings": baseline_rankings,
            "eligible": set(),
            "changed": set(),
        }
    ]
    if not primary_pool_predictions.empty:
        for probe_scope, scope_predictions in primary_pool_predictions.groupby(
            "scope", sort=False
        ):
            learned_maps = score_maps_from_predictions(scope_predictions)
            for depth in rerank_depths:
                rankings, eligible, changed = rerank_native_prefix(
                    artifact, learned_maps, depth=depth
                )
                variants.append(
                    {
                        "variant": (
                            f"oof_lr_{probe_scope}_{PRIMARY_PROBE_FEATURE}_depth{depth}"
                        ),
                        "method": "oof_logistic_regression",
                        "depth": depth,
                        "probe_scope": str(probe_scope),
                        "rankings": rankings,
                        "eligible": eligible,
                        "changed": changed,
                    }
                )

    max_depth = max(rerank_depths)
    rows: list[dict[str, Any]] = []
    official_artifacts: dict[str, Any] = {}
    primary_query_metrics: dict[str, pd.DataFrame] = {}
    for variant in variants:
        name = str(variant["variant"])
        metrics = per_query_official_metrics(
            artifact, variant["rankings"], ground_truth
        )
        jaccard_columns = per_query_ranking_jaccard_columns(
            baseline_rankings,
            variant["rankings"],
        )
        query_rows = np.asarray(
            [
                artifact.key_to_index[official_key(query_id)]
                for query_id in metrics["query_id"]
            ],
            dtype=np.int64,
        )
        for column, values in jaccard_columns.items():
            metrics[column] = values[query_rows]
        if variant["depth"] == 10 and not np.array_equal(
            np.sort(variant["rankings"][:, :10], axis=1),
            np.sort(baseline_rankings[:, :10], axis=1),
        ):
            raise AssertionError(f"Top-10 rerank changed candidate membership: {name}")
        if variant["depth"] == 10 and not np.array_equal(
            metrics["negative_recall10"].to_numpy(),
            baseline_metrics["negative_recall10"].to_numpy(),
        ):
            raise AssertionError(f"Top-10 rerank changed NegRecall@10: {name}")

        retain_query_metrics = name == "native" or (
            variant["depth"] == max_depth
            and variant["method"] == "oof_logistic_regression"
        )
        primary_official = name == "native" or (
            variant["depth"] == max_depth
            and (
                variant["method"] == "oof_logistic_regression"
                and variant["probe_scope"] == PRIMARY_PROBE_SCOPE
            )
        )
        if retain_query_metrics:
            per_query = metrics.copy()
            per_query["model"] = artifact.spec.name
            per_query["variant"] = name
            per_query["group_id"] = per_query["query_id"].map(query_to_group)
            primary_query_metrics[name] = per_query
        official_metrics: dict[str, Any] | None = None
        if official_eval_all or primary_official:
            safe_name = _safe_variant_name(name)
            metrics_path = model_dir / f"{safe_name}_official_metrics.csv"
            with tempfile.TemporaryDirectory(
                prefix="counterfactual_official_eval_"
            ) as temporary_dir:
                results_path = (
                    model_dir / f"{safe_name}.json"
                    if primary_official
                    else Path(temporary_dir) / f"{safe_name}.json"
                )
                write_official_results(results_path, artifact, variant["rankings"])
                official_metrics = invoke_official_evaluator(
                    pinpoint_root=pinpoint_root,
                    ground_truth=ground_truth_path,
                    results_path=results_path,
                    metrics_path=metrics_path,
                )
            full_map = float(metrics["ap10"].mean())
            full_negative_recall = float(metrics["negative_recall10"].mean())
            if not math.isclose(
                float(official_metrics["mAP@10"]), full_map, abs_tol=5.1e-5
            ) or not math.isclose(
                float(official_metrics["NegRecall@10"]),
                full_negative_recall,
                abs_tol=5.1e-5,
            ):
                raise AssertionError(
                    f"internal metrics disagree with official evaluator for {name}: "
                    f"internal=({full_map}, {full_negative_recall}), "
                    f"official=({official_metrics['mAP@10']}, "
                    f"{official_metrics['NegRecall@10']})"
                )
            official_artifacts[name] = {
                "results": str(results_path) if primary_official else None,
                "metrics": str(metrics_path),
            }

        baseline_cohorts = {
            row["cohort"]: row
            for row in cohort_metric_rows(
                baseline_metrics,
                pilot_query_ids=pilot_query_ids,
                strict_query_ids=strict_query_ids,
                object_pair_query_ids=object_pair_query_ids,
            )
        }
        for cohort_row in cohort_metric_rows(
            metrics,
            pilot_query_ids=pilot_query_ids,
            strict_query_ids=strict_query_ids,
            object_pair_query_ids=object_pair_query_ids,
        ):
            cohort = cohort_row["cohort"]
            displayed_cohort = (
                "official_full_native_7635"
                if cohort == "official_full_7635" and name == "native"
                else "official_full_hybrid_pilot_only_rerank"
                if cohort == "official_full_7635"
                else cohort
            )
            row: dict[str, Any] = {
                **cohort_row,
                "model": artifact.spec.name,
                "variant": name,
                "method": variant["method"],
                "rerank_depth": variant["depth"],
                "probe_scope": variant["probe_scope"],
                "cohort": displayed_cohort,
                "profile_coverage": "none_native"
                if name == "native"
                else "pilot_strict_subset_only",
                "rerank_eligible_query_count": len(variant["eligible"]),
                "rerank_changed_query_count": len(variant["changed"]),
                "official_queries_unchanged_by_rerank": (
                    len(ground_truth) - len(variant["eligible"])
                    if name != "native"
                    else len(ground_truth)
                ),
                "delta_mAP@10": (
                    cohort_row["mAP@10"] - baseline_cohorts[cohort]["mAP@10"]
                ),
                "delta_NegRecall@10": (
                    cohort_row["NegRecall@10"]
                    - baseline_cohorts[cohort]["NegRecall@10"]
                ),
                "official_evaluator_invoked": official_metrics is not None,
                "official_mAP@10_rounded": (
                    official_metrics.get("mAP@10") if official_metrics else None
                ),
                "official_NegRecall@10_rounded": (
                    official_metrics.get("NegRecall@10") if official_metrics else None
                ),
            }
            if cohort != "official_full_7635":
                cohort_ids = (
                    pilot_query_ids
                    if cohort == "pilot_500_all_queries"
                    else strict_query_ids
                    if cohort == "pilot_strict_negative_eligible"
                    else object_pair_query_ids
                )
                for metric_name in ("mAP@10", "NegRecall@10"):
                    ci = _paired_group_delta_ci(
                        baseline_metrics,
                        metrics,
                        query_ids=cohort_ids,
                        query_to_group=query_to_group,
                        metric=metric_name,
                        replicates=bootstrap_replicates,
                        seed=seed
                        + int(stable_hash([name, cohort, metric_name])[:8], 16),
                    )
                    prefix = "map10" if metric_name == "mAP@10" else "negrecall10"
                    row[f"paired_group_macro_delta_{prefix}"] = ci["estimate"]
                    row[f"paired_group_macro_delta_{prefix}_ci_low"] = ci["ci_low"]
                    row[f"paired_group_macro_delta_{prefix}_ci_high"] = ci["ci_high"]
            rows.append(row)
    result_frame = pd.DataFrame(rows)
    result_frame.to_csv(model_dir / "rerank_metrics_all_cohorts.csv", index=False)
    if primary_query_metrics:
        pd.concat(primary_query_metrics.values(), ignore_index=True).to_parquet(
            model_dir / "primary_rerank_metrics_by_query.parquet",
            index=False,
            compression="zstd",
        )
    return (
        result_frame,
        {"official_artifacts": official_artifacts},
        primary_query_metrics,
    )


def paired_model_profile_statistics(
    baseline_values: dict[str, pd.DataFrame],
    hard_values: dict[str, pd.DataFrame],
    *,
    baseline_name: str,
    hard_name: str,
    pool_name: str,
    bootstrap_replicates: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for key in sorted(set(baseline_values) & set(hard_values)):
        baseline = baseline_values[key].rename(
            columns={"difference": "baseline_separation"}
        )
        hard = hard_values[key].rename(columns={"difference": "hard_separation"})
        paired = baseline.merge(
            hard,
            on=["group_id", "query_id"],
            validate="one_to_one",
        )
        paired["hard_minus_baseline"] = (
            paired["hard_separation"] - paired["baseline_separation"]
        )
        contrast, dimension = key.split("::", 1)
        summary = group_macro_bootstrap(
            paired[["group_id", "query_id", "hard_minus_baseline"]],
            value_column="hard_minus_baseline",
            replicates=bootstrap_replicates,
            seed=seed + int(stable_hash([baseline_name, hard_name, key])[:8], 16),
        )
        rows.append(
            {
                "baseline_model": baseline_name,
                "hard_model": hard_name,
                "pool_name": pool_name,
                "contrast": contrast,
                "dimension": dimension,
                **summary,
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["holm_p"] = math.nan
        for _, positions in frame.groupby("contrast", sort=False).groups.items():
            frame.loc[positions, "holm_p"] = holm_adjust(
                frame.loc[positions, "random_sign_p_two_sided"].tolist()
            )
    return frame


def paired_model_probe_statistics(
    baseline_metrics: dict[tuple[str, str], pd.DataFrame],
    hard_metrics: dict[tuple[str, str], pd.DataFrame],
    *,
    baseline_name: str,
    hard_name: str,
    pool_name: str,
    bootstrap_replicates: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for scope, feature in sorted(set(baseline_metrics) & set(hard_metrics)):
        if scope not in CROSS_MODEL_FIXED_PROBE_SCOPES:
            continue
        baseline = baseline_metrics[(scope, feature)]
        hard = hard_metrics[(scope, feature)]
        if set(baseline["query_id"]) != set(hard["query_id"]):
            raise ValueError(
                f"shared-pool probe cohort differs across models for {scope}/{feature}"
            )
        paired = baseline.merge(
            hard,
            on=["group_id", "query_id"],
            suffixes=("_baseline", "_hard"),
            validate="one_to_one",
        )
        if (
            not paired["candidate_cohort_fingerprint_baseline"]
            .eq(paired["candidate_cohort_fingerprint_hard"])
            .all()
        ):
            raise ValueError(
                f"shared-pool candidate/label cohort differs across models for "
                f"{scope}/{feature}"
            )
        for metric in ("query_ap", "query_pairwise_auc"):
            paired["hard_minus_baseline"] = (
                paired[f"{metric}_hard"] - paired[f"{metric}_baseline"]
            )
            summary = group_macro_bootstrap(
                paired[["group_id", "query_id", "hard_minus_baseline"]],
                value_column="hard_minus_baseline",
                replicates=bootstrap_replicates,
                seed=seed
                + int(
                    stable_hash([baseline_name, hard_name, scope, feature, metric])[:8],
                    16,
                ),
            )
            rows.append(
                {
                    "baseline_model": baseline_name,
                    "hard_model": hard_name,
                    "pool_name": pool_name,
                    "scope": scope,
                    "feature": feature,
                    "metric": metric,
                    **summary,
                }
            )
    return pd.DataFrame(rows)


def paired_model_incremental_probe_did(
    baseline_metrics: dict[tuple[str, str], pd.DataFrame],
    hard_metrics: dict[tuple[str, str], pd.DataFrame],
    *,
    baseline_name: str,
    hard_name: str,
    pool_name: str,
    bootstrap_replicates: int,
    seed: int,
) -> pd.DataFrame:
    """Compare feature gain across models on identical shared candidates."""

    rows = []
    scopes = sorted(
        (
            {scope for scope, _ in baseline_metrics}
            & {scope for scope, _ in hard_metrics}
        )
        & set(CROSS_MODEL_FIXED_PROBE_SCOPES)
    )
    for scope in scopes:
        for enhanced_feature, control_feature in FEATURE_COMPARISONS:
            required = (
                (scope, enhanced_feature),
                (scope, control_feature),
            )
            if any(
                key not in baseline_metrics or key not in hard_metrics
                for key in required
            ):
                continue
            model_increments: dict[str, pd.DataFrame] = {}
            for model_role, metrics in (
                ("baseline", baseline_metrics),
                ("hard", hard_metrics),
            ):
                enhanced = metrics[(scope, enhanced_feature)]
                control = metrics[(scope, control_feature)]
                if set(enhanced["query_id"]) != set(control["query_id"]):
                    raise ValueError(
                        f"shared feature DID changed {model_role} cohort for "
                        f"{scope}/{enhanced_feature}/{control_feature}"
                    )
                paired = enhanced.merge(
                    control,
                    on=["group_id", "query_id"],
                    suffixes=("_enhanced", "_control"),
                    validate="one_to_one",
                )
                if (
                    not paired["candidate_cohort_fingerprint_enhanced"]
                    .eq(paired["candidate_cohort_fingerprint_control"])
                    .all()
                ):
                    raise ValueError(
                        f"shared feature DID changed {model_role} candidate/label "
                        f"cohort for {scope}/{enhanced_feature}/{control_feature}"
                    )
                model_increments[model_role] = paired
            if set(model_increments["baseline"]["query_id"]) != set(
                model_increments["hard"]["query_id"]
            ):
                raise ValueError(
                    f"shared feature DID differs across models for {scope}/"
                    f"{enhanced_feature}/{control_feature}"
                )
            for metric in ("query_ap", "query_pairwise_auc"):
                baseline_increment = model_increments["baseline"][
                    [
                        "group_id",
                        "query_id",
                        "candidate_cohort_fingerprint_enhanced",
                    ]
                ].copy()
                baseline_increment = baseline_increment.rename(
                    columns={
                        "candidate_cohort_fingerprint_enhanced": (
                            "candidate_cohort_fingerprint"
                        )
                    }
                )
                baseline_increment["baseline_increment"] = (
                    model_increments["baseline"][f"{metric}_enhanced"]
                    - model_increments["baseline"][f"{metric}_control"]
                )
                hard_increment = model_increments["hard"][
                    [
                        "group_id",
                        "query_id",
                        "candidate_cohort_fingerprint_enhanced",
                    ]
                ].copy()
                hard_increment = hard_increment.rename(
                    columns={
                        "candidate_cohort_fingerprint_enhanced": (
                            "candidate_cohort_fingerprint"
                        )
                    }
                )
                hard_increment["hard_increment"] = (
                    model_increments["hard"][f"{metric}_enhanced"]
                    - model_increments["hard"][f"{metric}_control"]
                )
                did = baseline_increment.merge(
                    hard_increment,
                    on=["group_id", "query_id"],
                    suffixes=("_baseline", "_hard"),
                    validate="one_to_one",
                )
                if (
                    not did["candidate_cohort_fingerprint_baseline"]
                    .eq(did["candidate_cohort_fingerprint_hard"])
                    .all()
                ):
                    raise ValueError(
                        f"shared feature DID candidate/label cohort differs across "
                        f"models for {scope}/{enhanced_feature}/{control_feature}"
                    )
                did["hard_minus_baseline_increment"] = (
                    did["hard_increment"] - did["baseline_increment"]
                )
                summary = group_macro_bootstrap(
                    did[
                        [
                            "group_id",
                            "query_id",
                            "hard_minus_baseline_increment",
                        ]
                    ],
                    value_column="hard_minus_baseline_increment",
                    replicates=bootstrap_replicates,
                    seed=seed
                    + int(
                        stable_hash(
                            [
                                baseline_name,
                                hard_name,
                                scope,
                                enhanced_feature,
                                control_feature,
                                metric,
                                "incremental_did",
                            ]
                        )[:8],
                        16,
                    ),
                )
                rows.append(
                    {
                        "baseline_model": baseline_name,
                        "hard_model": hard_name,
                        "pool_name": pool_name,
                        "scope": scope,
                        "enhanced_feature": enhanced_feature,
                        "control_feature": control_feature,
                        "metric": metric,
                        **summary,
                    }
                )
    return pd.DataFrame(rows)


def paired_rerank_delta_of_differences(
    baseline_metrics: dict[str, pd.DataFrame],
    hard_metrics: dict[str, pd.DataFrame],
    *,
    baseline_name: str,
    hard_name: str,
    coverage_frame: pd.DataFrame,
    bootstrap_replicates: int,
    seed: int,
) -> pd.DataFrame:
    if "native" not in baseline_metrics or "native" not in hard_metrics:
        return pd.DataFrame()
    baseline_native = baseline_metrics["native"]
    hard_native = hard_metrics["native"]
    strict_ids = {
        normalize_query_id(row.query_id)
        for row in coverage_frame.itertuples()
        if row.has_any_strict
    }
    object_pair_ids = {
        normalize_query_id(row.query_id)
        for row in coverage_frame.itertuples()
        if row.has_object_pair
    }
    pilot_ids = {
        normalize_query_id(row.query_id) for row in coverage_frame.itertuples()
    }
    cohorts = {
        "pilot_500_all_queries": pilot_ids,
        "pilot_strict_negative_eligible": strict_ids,
        "pilot_object_add_substitute_pair": object_pair_ids,
    }
    variants = sorted((set(baseline_metrics) & set(hard_metrics)) - {"native"})
    rows = []
    for variant in variants:
        joined = (
            baseline_native[["query_id", "group_id", "ap10", "negative_recall10"]]
            .merge(
                baseline_metrics[variant][["query_id", "ap10", "negative_recall10"]],
                on="query_id",
                suffixes=("_baseline_native", "_baseline_rerank"),
                validate="one_to_one",
            )
            .merge(
                hard_native[["query_id", "ap10", "negative_recall10"]],
                on="query_id",
                validate="one_to_one",
            )
            .rename(
                columns={
                    "ap10": "ap10_hard_native",
                    "negative_recall10": "negative_recall10_hard_native",
                }
            )
            .merge(
                hard_metrics[variant][["query_id", "ap10", "negative_recall10"]],
                on="query_id",
                validate="one_to_one",
            )
            .rename(
                columns={
                    "ap10": "ap10_hard_rerank",
                    "negative_recall10": "negative_recall10_hard_rerank",
                }
            )
        )
        for cohort, query_ids in cohorts.items():
            selected = joined.loc[joined["query_id"].isin(query_ids)].copy()
            for metric, column in (
                ("mAP@10", "ap10"),
                ("NegRecall@10", "negative_recall10"),
            ):
                selected["delta_of_deltas"] = (
                    selected[f"{column}_hard_rerank"]
                    - selected[f"{column}_hard_native"]
                    - selected[f"{column}_baseline_rerank"]
                    + selected[f"{column}_baseline_native"]
                )
                summary = group_macro_bootstrap(
                    selected[["group_id", "query_id", "delta_of_deltas"]],
                    value_column="delta_of_deltas",
                    replicates=bootstrap_replicates,
                    seed=seed
                    + int(
                        stable_hash(
                            [baseline_name, hard_name, variant, cohort, metric]
                        )[:8],
                        16,
                    ),
                )
                rows.append(
                    {
                        "baseline_model": baseline_name,
                        "hard_model": hard_name,
                        "variant": variant,
                        "cohort": cohort,
                        "metric": metric,
                        **summary,
                    }
                )
    return pd.DataFrame(rows)


def cohort_representativeness_audit(
    ground_truth: pd.DataFrame,
    coverage_frame: pd.DataFrame,
    artifacts: Sequence[ModelArtifacts],
) -> pd.DataFrame:
    pilot_ids = {
        normalize_query_id(value) for value in coverage_frame["query_id"].tolist()
    }
    strict_ids = {
        normalize_query_id(row.query_id)
        for row in coverage_frame.itertuples()
        if row.has_any_strict
    }
    object_pair_ids = {
        normalize_query_id(row.query_id)
        for row in coverage_frame.itertuples()
        if row.has_object_pair
    }
    cohorts = {
        "official_full_7635": set(ground_truth.index),
        "pilot_500_all_queries": pilot_ids,
        "pilot_strict_negative_eligible": strict_ids,
        "pilot_object_add_substitute_pair": object_pair_ids,
    }
    rows = []
    native_metrics = {
        artifact.spec.name: per_query_official_metrics(
            artifact,
            artifact.top_indices,
            ground_truth,
        ).set_index("query_id")
        for artifact in artifacts
    }
    for cohort, query_ids in cohorts.items():
        selected = ground_truth.loc[ground_truth.index.intersection(sorted(query_ids))]
        base_values = {
            "cohort": cohort,
            "query_count": len(selected),
            "mean_unique_positive_count": float(
                selected["positive_clean"].map(len).mean()
            ),
            "mean_unique_official_negative_count": float(
                selected["negative_clean"].map(len).mean()
            ),
            "mean_raw_official_negative_count": float(
                selected["negative_candidates"]
                .map(lambda value: len(_raw_identifier_values(value)))
                .mean()
            ),
        }
        rows.append({"model": "ground_truth_only", **base_values})
        for artifact in artifacts:
            positive_top10 = []
            positive_top50 = []
            negative_top10 = []
            negative_top50 = []
            for query_id, gt_row in selected.iterrows():
                row = artifact.key_to_index[official_key(query_id)]
                top_ids = artifact.signatures[artifact.top_indices[row]].astype(str)
                positives = set(gt_row["positive_clean"])
                negatives = set(gt_row["negative_clean"]) - positives
                positive_top10.append(any(value in positives for value in top_ids[:10]))
                positive_top50.append(any(value in positives for value in top_ids[:50]))
                negative_top10.append(any(value in negatives for value in top_ids[:10]))
                negative_top50.append(any(value in negatives for value in top_ids[:50]))
            rows.append(
                {
                    "model": artifact.spec.name,
                    **base_values,
                    "native_mAP@10": float(
                        native_metrics[artifact.spec.name]
                        .loc[selected.index, "ap10"]
                        .mean()
                    ),
                    "native_NegRecall@10": float(
                        native_metrics[artifact.spec.name]
                        .loc[selected.index, "negative_recall10"]
                        .mean()
                    ),
                    "query_fraction_with_positive_top10": float(
                        np.mean(positive_top10)
                    ),
                    "query_fraction_with_positive_top50": float(
                        np.mean(positive_top50)
                    ),
                    "query_fraction_with_official_negative_top10": float(
                        np.mean(negative_top10)
                    ),
                    "query_fraction_with_official_negative_top50": float(
                        np.mean(negative_top50)
                    ),
                }
            )
    return pd.DataFrame(rows)


def cohort_composition_frame(
    ground_truth: pd.DataFrame,
    coverage_frame: pd.DataFrame,
) -> pd.DataFrame:
    pilot_ids = {
        normalize_query_id(value) for value in coverage_frame["query_id"].tolist()
    }
    strict_ids = {
        normalize_query_id(row.query_id)
        for row in coverage_frame.itertuples()
        if row.has_any_strict
    }
    object_pair_ids = {
        normalize_query_id(row.query_id)
        for row in coverage_frame.itertuples()
        if row.has_object_pair
    }
    cohorts = {
        "official_full_7635": set(ground_truth.index),
        "pilot_500_all_queries": pilot_ids,
        "pilot_strict_negative_eligible": strict_ids,
        "pilot_object_add_substitute_pair": object_pair_ids,
    }
    rows = []
    for cohort, query_ids in cohorts.items():
        selected = ground_truth.loc[ground_truth.index.intersection(sorted(query_ids))]
        dimensions: dict[str, pd.Series] = {
            "query_category": selected.get(
                "query_category", pd.Series("unknown", index=selected.index)
            )
            .fillna("unknown")
            .astype(str),
            "length_category": selected.get(
                "length_category", pd.Series("unknown", index=selected.index)
            )
            .fillna("unknown")
            .astype(str),
            "reference_count": selected.apply(
                lambda row: (
                    1
                    + int(
                        row.get("query_image_signature2") is not None
                        and str(row.get("query_image_signature2")).lower()
                        not in {"", "none", "nan"}
                    )
                ),
                axis=1,
            ).astype(str),
        }
        for dimension, values in dimensions.items():
            counts = values.value_counts(dropna=False)
            for value, count in counts.items():
                rows.append(
                    {
                        "cohort": cohort,
                        "dimension": dimension,
                        "value": value,
                        "query_count": int(count),
                        "query_fraction": float(count / len(selected)),
                    }
                )
    return pd.DataFrame(rows)


def _format_markdown_table(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    max_rows: int = 20,
) -> str:
    if frame.empty:
        return "（无结果）"
    selected = frame.loc[:, [column for column in columns if column in frame]].head(
        max_rows
    )
    headers = selected.columns.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in selected.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append("NA" if not np.isfinite(value) else f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_model_decision_summary(
    text_sensitivity: pd.DataFrame,
    probe_results: pd.DataFrame,
    paired_feature_results: pd.DataFrame,
    rerank_results: pd.DataFrame,
    *,
    primary_depth: int,
) -> pd.DataFrame:
    if text_sensitivity.empty:
        return pd.DataFrame()
    summary = text_sensitivity.loc[
        text_sensitivity["subset"].eq("strict"),
        [
            "model",
            "pair_balanced_mean_target_delta",
            "pair_balanced_target_raw_win_rate",
        ],
    ].copy()
    summary.insert(
        1,
        "analysis_mode",
        "simple_no_ml" if probe_results.empty else "simple_plus_oof_probe",
    )
    if not probe_results.empty:
        summary.insert(2, "probe_candidate_pool", "model_native_top50")
        selected_probes = probe_results.loc[
            probe_results["scope"].eq(PRIMARY_PROBE_SCOPE)
            & probe_results["feature"].isin(
                [
                    "source_similarity_strict",
                    "d_available_only",
                    "source_plus_d_available",
                    "source_plus_masked_6d",
                ]
            )
            & probe_results.get("status", "ok").eq("ok")
        ]
        for metric in (
            "weighted_auroc",
            "weighted_auprc",
            "query_macro_pairwise_auc",
        ):
            pivot = selected_probes.pivot(
                index="model", columns="feature", values=metric
            ).add_prefix(f"{metric}::")
            summary = summary.merge(pivot.reset_index(), on="model", how="left")
    if not paired_feature_results.empty:
        gain = paired_feature_results.loc[
            paired_feature_results["scope"].eq(PRIMARY_PROBE_SCOPE)
            & paired_feature_results["enhanced_feature"].eq("source_plus_d_available")
            & paired_feature_results["control_feature"].eq("source_similarity_strict")
            & paired_feature_results["metric"].eq("query_pairwise_auc"),
            ["model", "estimate", "ci_low", "ci_high"],
        ].rename(
            columns={
                "estimate": "paired_delta_query_pairwise_auc_sim_plus_d",
                "ci_low": "paired_delta_query_pairwise_auc_ci_low",
                "ci_high": "paired_delta_query_pairwise_auc_ci_high",
            }
        )
        summary = summary.merge(gain, on="model", how="left")
    if not rerank_results.empty:
        full_native = rerank_results.loc[
            rerank_results["cohort"].eq("official_full_native_7635")
            & rerank_results["variant"].eq("native"),
            ["model", "mAP@10", "NegRecall@10"],
        ].rename(
            columns={
                "mAP@10": "official_full_native_mAP@10",
                "NegRecall@10": "official_full_native_NegRecall@10",
            }
        )
        summary = summary.merge(full_native, on="model", how="left")
        pilot = rerank_results.loc[
            rerank_results["cohort"].eq("pilot_strict_negative_eligible")
            & (
                rerank_results["variant"].eq("native")
                | (
                    rerank_results["method"].eq("oof_logistic_regression")
                    & rerank_results["probe_scope"].eq(PRIMARY_PROBE_SCOPE)
                    & rerank_results["rerank_depth"].eq(primary_depth)
                )
            )
        ].copy()
        pilot["setting"] = np.where(
            pilot["variant"].eq("native"), "native", "profile_oof_rerank"
        )
        for metric in ("mAP@10", "NegRecall@10"):
            pivot = pilot.pivot(
                index="model", columns="setting", values=metric
            ).add_prefix(f"pilot_strict_{metric}::")
            summary = summary.merge(pivot.reset_index(), on="model", how="left")
    return summary


def write_analysis_report(
    path: Path,
    *,
    coverage_summary: dict[str, Any],
    candidate_summaries: Sequence[dict[str, Any]],
    text_sensitivity_results: pd.DataFrame,
    target_exposure_audit: pd.DataFrame,
    decision_summary: pd.DataFrame,
    pair_decision_summary: pd.DataFrame,
    probe_results: pd.DataFrame,
    paired_feature_results: pd.DataFrame,
    rerank_results: pd.DataFrame,
    cross_model_rerank_results: pd.DataFrame,
    paired_results: pd.DataFrame,
    native_jaccard_summary: pd.DataFrame,
    shared_probe_pair_results: pd.DataFrame,
    shared_incremental_probe_did: pd.DataFrame,
    rerank_delta_of_differences: pd.DataFrame,
) -> None:
    primary_probe = (
        probe_results.loc[
            probe_results.get("scope", pd.Series(dtype=object)).isin(
                RERANK_PROBE_SCOPES
            )
            & probe_results.get("feature", pd.Series(dtype=object)).isin(
                [
                    "source_similarity_strict",
                    "positive_ensemble_strict",
                    "source_plus_d_available",
                    PRIMARY_PROBE_FEATURE,
                    "candidate_shuffled_placebo",
                    "mask_only",
                ]
            )
        ]
        if not probe_results.empty
        else probe_results
    )
    primary_rerank = (
        rerank_results.loc[
            rerank_results.get("cohort", pd.Series(dtype=object)).isin(
                [
                    "official_full_native_7635",
                    "official_full_hybrid_pilot_only_rerank",
                    "pilot_strict_negative_eligible",
                ]
            )
            & (
                rerank_results.get("variant", pd.Series(dtype=object)).eq("native")
                | rerank_results.get("variant", pd.Series(dtype=object)).str.contains(
                    PRIMARY_PROBE_FEATURE, regex=False
                )
            )
        ]
        if not rerank_results.empty
        else rerank_results
    )
    rank_violation_probes = (
        probe_results.loc[
            probe_results.get("scope", pd.Series(dtype=object)).isin(
                RANK_VIOLATION_PROBE_SCOPES
            )
            & probe_results.get("feature", pd.Series(dtype=object)).isin(
                [
                    "source_similarity_strict",
                    "d_available_only",
                    "source_plus_d_available",
                    "source_plus_masked_6d",
                ]
            )
        ]
        if not probe_results.empty
        else probe_results
    )
    target_exposure_primary = (
        target_exposure_audit.loc[
            target_exposure_audit.get("dimension", pd.Series(dtype=object)).isin(
                ("d_available", *RESPONSE_COLUMNS)
            )
        ]
        if not target_exposure_audit.empty
        else target_exposure_audit
    )
    oof_enabled = not probe_results.empty
    decision_probe_note = (
        "上表的 probe 来自各模型自己的 native Top-50；跨模型的 OOF 结论只看固定 "
        "shared-union 候选上的配对 DID。"
        if oof_enabled
        else "本次按 simple_no_ml 口径运行：没有训练 LogisticRegression，也没有使用 "
        "learned reranker；只报告原始检索、直接统计和跨模型 q+ similarity rerank。"
    )
    probe_control_note = (
        "- 可选 probe 按真实 V7 group 做 OOF，所有预处理和 LogisticRegression 均在 fold 内拟合。"
        if oof_enabled
        else "- 本次关闭全部 OOF probe/learned reranker；旧运行遗留文件不进入本报告汇总。"
    )
    probe_section = (
        f"""## 可选 OOF probe（本次已启用）

{_format_markdown_table(primary_probe, ["model", "scope", "feature", "query_count", "group_count", "weighted_auroc", "weighted_auprc", "query_macro_ap", "query_macro_pairwise_auc"])}

同一 cohort、同一 OOF folds 的 feature 配对增益：

{_format_markdown_table(paired_feature_results, ["model", "pool_name", "scope", "enhanced_feature", "control_feature", "metric", "estimate", "ci_low", "ci_high", "query_count", "group_count"])}

只看排在 best positive 之上的 rank-violating candidates：

{_format_markdown_table(rank_violation_probes, ["model", "scope", "feature", "query_count", "group_count", "weighted_auroc", "weighted_auprc", "query_macro_ap", "query_macro_pairwise_auc"])}
"""
        if oof_enabled
        else """## 机器学习诊断

本次默认关闭 OOF LogisticRegression、AUROC/AUPRC probe 和 learned reranker。
"""
    )
    shared_probe_section = (
        f"""shared-union pool 上的 Hard − baseline OOF probe 差：

{_format_markdown_table(shared_probe_pair_results, ["baseline_model", "hard_model", "scope", "feature", "metric", "estimate", "ci_low", "ci_high", "query_count", "group_count"])}

shared-union pool 上的 profile 增量 DID：

{_format_markdown_table(shared_incremental_probe_did, ["baseline_model", "hard_model", "scope", "enhanced_feature", "control_feature", "metric", "estimate", "ci_low", "ci_high", "query_count", "group_count"])}

rank-violating scope 由各模型自身分数决定 eligible query，因此不做跨模型伪配对。

Hard 与 baseline 的 learned-rerank 改善量之差：

{_format_markdown_table(rerank_delta_of_differences, ["baseline_model", "hard_model", "variant", "cohort", "metric", "estimate", "ci_low", "ci_high", "query_count", "group_count"])}
"""
        if oof_enabled
        else "跨模型模型差仅报告直接 response separation 与原始 Top-K Jaccard；不计算 OOF/DID。"
    )
    text = f"""# Counterfactual response profile experiment

## 口径

- 官方 PinPoint 全量 **7,635 queries** 是主检索基准，报告原版 evaluator 的 mAP@10 与 NegRecall@10。
- V7 的 **{coverage_summary["group_count"]} groups / {coverage_summary["query_count"]} queries** 是官方 query 的子集，仅它们有生成并审核过的反事实 negative text，因此用于 response profile、probe 和诊断性 rerank。
- 两者不是二选一：全量回答“模型最终检索怎样”，500 组子集回答“模型为何这样，以及对 negative edit 如何响应”。

## 决策总表

{_format_markdown_table(decision_summary, ["model", "analysis_mode", "pair_balanced_mean_target_delta", "pair_balanced_target_raw_win_rate", "official_full_native_mAP@10", "official_full_native_NegRecall@10", "pilot_strict_mAP@10::native", "pilot_strict_NegRecall@10::native", "probe_candidate_pool", "weighted_auroc::source_similarity_strict", "weighted_auroc::d_available_only", "weighted_auroc::source_plus_d_available", "paired_delta_query_pairwise_auc_sim_plus_d"], max_rows=20)}

{decision_probe_note}

{_format_markdown_table(pair_decision_summary, ["baseline_model", "hard_model", "scope", "enhanced_feature", "control_feature", "metric", "estimate", "ci_low", "ci_high", "query_count", "group_count"]) if oof_enabled else ""}

## 稀疏 negative 规则

- strict 主分析只使用 ADD / SUBSTITUTE；DELETE 不进入主 response。
- 每条 query 有多少 accepted negative 就用多少：先对同一类别内文本求均值，再对该 query 实际存在的类别等权平均。
- 缺失类别保持 NaN，并带显式 mask；不填 0，不要求六类齐全。
- 有至少一类 strict negative 的 query：**{coverage_summary["query_with_any_strict_count"]}**；完整 6D 仅 **{coverage_summary["complete_6d_query_count"]}**，只报告覆盖率，不据此训练 probe。
- 固定 object_add + object_substitute 二维 cohort 是预先指定的补充验证。

## 防止误读的控制

- 所有官方 positives 都作为 multi-positive 标签；Top-K 外 positive 只追加到诊断训练集，不加入 rerank 池。
- 官方 negatives 与未标注 Top-K competitors 分开；未标注候选不被称作语义错误。
{probe_control_note}
- positive paraphrase ensemble、mask-only、candidate-shuffle、category-permutation 均作为 placebo/control。
- category-permutation 固定原有 missingness pattern，只在该 query 已出现的 edit 类别之间重分配 response；其 value 与显式命名的 permuted mask 列始终配套，mask 数值本身不随机置换。
- shared-union pool 用于 baseline/hard 模型的固定候选公平比较；native pool 用于真实运行指标。
- negative-response separation 仍是 target-informed 诊断；跨模型 rerank 完全不读取 V7 negative text，只使用两个现有模型的正常 q+ embedding。
- cross-model rerank 固定使用 retriever 的 native Top-50；reranker 只提供自己的正常 q+ cosine score，不读取 negative text。

## 候选构建

{_format_markdown_table(pd.DataFrame(candidate_summaries), ["model", "pool_name", "query_count", "candidate_row_count", "retrieved_pool_row_count", "appended_positive_row_count"])}

## 原有 positive-vs-negative-text Delta / 胜率（同一批 q- 现场复算）

{_format_markdown_table(text_sensitivity_results.loc[text_sensitivity_results.get("subset", pd.Series(dtype=object)).eq("strict")] if not text_sensitivity_results.empty else text_sensitivity_results, ["model", "subset", "group_count", "instruction_count", "negative_count", "pair_balanced_mean_target_delta", "mean_target_delta_ci_low", "mean_target_delta_ci_high", "pair_balanced_target_raw_win_rate", "target_raw_win_rate_ci_low", "target_raw_win_rate_ci_high"])}

## Target exposure 泄漏审计

V7 q- 的生成/审核看过 validated positives。下表在同一 query 内比较这些 positives 与生成阶段
未列入 validated set 的其它官方 positives；它能暴露明显的 target-specific response，但不能替代
真正 target-blind 的独立测试集。

{_format_markdown_table(target_exposure_primary, ["model", "dimension", "first_mean", "second_mean", "estimate", "ci_low", "ci_high", "query_count", "group_count", "holm_p"], max_rows=50)}

{probe_section}

## Native 检索核心结果

{_format_markdown_table(primary_rerank, ["model", "variant", "cohort", "query_count", "mAP@10", "delta_mAP@10", "NegRecall@10", "delta_NegRecall@10", "Top1JaccardVsNative", "Top5JaccardVsNative", "Top10JaccardVsNative", "Top20JaccardVsNative", "Top50JaccardVsNative", "rerank_eligible_query_count"])}

## 跨模型 Top-50 rerank（不使用 negative text）

`retriever_model` 先从完整图库召回自己的 native Top-50；`reranker_model` 使用自己的
正常 positive query embedding 与自己的 candidate image embedding 计算 cosine score，只在
这 50 个 candidate 内重新排序。

{_format_markdown_table(cross_model_rerank_results, ["retriever_model", "reranker_model", "cohort", "query_count", "native_mAP@10", "mAP@10", "delta_mAP@10", "native_PositiveRecall@10", "PositiveRecall@10", "delta_PositiveRecall@10", "native_nDCG@10", "nDCG@10", "delta_nDCG@10", "native_NegRecall@10", "NegRecall@10", "delta_NegRecall@10", "Top1JaccardVsNative", "Top5JaccardVsNative", "Top10JaccardVsNative", "Top20JaccardVsNative", "Top50JaccardVsNative", "uses_negative_text"], max_rows=50)}

## Baseline / Hard 原始检索集合 Jaccard

Jaccard 只比较 Top-K 候选集合，不考虑集合内部顺序；同时报告更直观的
`mean_overlap_fraction = |A ∩ B| / K`。

{_format_markdown_table(native_jaccard_summary, ["baseline_model", "hard_model", "cohort", "k", "query_count", "mean_jaccard", "mean_overlap_fraction", "baseline_positive_recall", "hard_positive_recall", "delta_positive_recall", "baseline_positive_hit", "hard_positive_hit", "delta_positive_hit", "baseline_negative_recall", "hard_negative_recall", "delta_negative_recall", "baseline_ndcg", "hard_ndcg", "delta_ndcg", "bootstrap_unit"], max_rows=100)}

## 固定 shared-pool 模型差

{_format_markdown_table(paired_results, ["baseline_model", "hard_model", "contrast", "dimension", "estimate", "ci_low", "ci_high", "query_count", "group_count", "holm_p"])}

{shared_probe_section}

## 输出说明

- `negative_text_coverage_*.csv/json`：每条 query 的稀疏维度覆盖率。
- `target_exposure_audit_all_models.csv`：validated 与 unseen official positives 的同-query response 敏感性。
- `native_ranking_jaccard_summary.csv`：baseline/hard 原始 Top-K 候选集合重合度。
- `native_topk_retrieval_summary.csv`：同表的 Recall/HitRate/NegRecall/nDCG 导向别名。
- `cross_model_rerank_results.csv`：baseline/hard 互相作为 retriever/reranker 的正式结果。
- `cross_model_rerank_by_query.parquet`：跨模型 rerank 的逐 query 指标与 Jaccard。
- `native_ranking_jaccard_by_query.csv`：逐 query、逐 K 的 overlap 与 Jaccard。
- `PAIR_DECISION_SUMMARY.csv`：shared-union 固定候选上的预注册 profile 增量 DID 主表。
- `<model>/candidate_profiles_native.parquet`：candidate-level response，不把缺失值补 0。
- `<model>/negative_text_sensitivity_summary.csv`：原有 target Delta / win rate 基线。
- `<model>/profile_dimension_stats.csv`：multi-positive 候选分离与 group bootstrap CI。
- `<model>/probe_results.csv`：按 group 的 OOF probe 和 placebo。
- `<model>/paired_feature_probe_differences.csv`：严格相同 cohort 上 sim→sim+D 的配对增益。
- `<model>/rerank/rerank_metrics_all_cohorts.csv`：全量、Pilot、eligible、固定二维 cohort。
- `shared_pool_comparisons/`：固定候选并集的模型配对比较。
- `shared_pool_incremental_feature_did.csv`：profile 相对 sim 的增量在 Hard 与 baseline 间的配对 DID。
"""
    path.write_text(text, encoding="utf-8")


def _parse_int_values(raw: str, *, option: str) -> tuple[int, ...]:
    values = tuple(
        dict.fromkeys(int(value.strip()) for value in raw.split(",") if value.strip())
    )
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{option} expects positive comma-separated integers")
    return values


def main() -> None:
    args = parse_args()
    if args.top_k != 50:
        raise ValueError(
            "this protocol requires the official Top-50 cache; set --top-k 50"
        )
    rerank_depths = _parse_int_values(args.rerank_depths, option="--rerank-depths")
    if max(rerank_depths) > args.top_k:
        raise ValueError("maximum rerank depth cannot exceed --top-k")
    if args.bootstrap_replicates < 0:
        raise ValueError("--bootstrap-replicates must be non-negative")
    for required in (
        args.input,
        args.ground_truth,
        args.pinpoint_root / "evaluate.py",
        args.pinpoint_root / "image_cache",
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    groups, all_v7_queries, loader_coverage = load_v7_dataset(
        args.input.resolve(), limit_groups=args.limit_groups
    )
    positive_queries = [
        query for query in all_v7_queries if query.query_kind == "positive"
    ]
    negatives = strict_negative_queries(all_v7_queries)
    if not negatives:
        raise ValueError(
            "the selected V7 groups contain no accepted ADD/SUBSTITUTE negative text"
        )
    coverage_frame = build_coverage_frame(groups, positive_queries, negatives)
    coverage_summary = write_coverage_outputs(coverage_frame, output_dir=output_dir)
    validated_positive_ids = validated_positive_ids_by_group(args.input.resolve())
    ground_truth = prepare_ground_truth(args.ground_truth.resolve())
    if len(ground_truth) != 7635:
        raise ValueError(
            "this experiment is defined for the official 7,635-query PinPoint ground truth; "
            f"found {len(ground_truth)}"
        )
    pilot_query_ids = {
        normalize_query_id(query.instruction_id) for query in positive_queries
    }
    missing_pilot = sorted(pilot_query_ids - set(ground_truth.index))
    if missing_pilot:
        raise KeyError(
            f"V7 pilot queries absent from official PinPoint: {missing_pilot[:5]}"
        )

    model_specs = resolve_model_specs(args)
    model_names = [spec.name for spec in model_specs]
    comparisons = parse_comparisons(args.comparison, model_names)
    artifacts = {
        spec.name: load_model_artifacts(
            spec,
            top_k=args.top_k,
            ground_truth_path=args.ground_truth.resolve(),
        )
        for spec in model_specs
    }
    fold_map = make_balanced_group_fold_map(
        coverage_frame,
        folds=args.folds,
        seed=args.seed,
    )
    pd.DataFrame(sorted(fold_map.items()), columns=["group_id", "fold"]).to_csv(
        output_dir / "group_fold_assignment.csv", index=False
    )
    if args.preflight_only:
        preflight = {
            "status": "ok",
            "official_query_count": len(ground_truth),
            "pilot_group_count": len(groups),
            "pilot_query_count": len(positive_queries),
            "strict_negative_text_count": len(negatives),
            "models": {
                name: {
                    "corpus_shape": list(artifact.corpus.shape),
                    "official_query_shape": list(artifact.positive_queries.shape),
                    "topk_shape": list(artifact.top_indices.shape),
                }
                for name, artifact in artifacts.items()
            },
        }
        write_json_atomic(output_dir / "preflight.json", preflight)
        print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)
        return

    if args.ranking_jaccard_only:
        if not comparisons:
            raise ValueError(
                "--ranking-jaccard-only requires at least one --comparison"
            )
        query_frames: list[pd.DataFrame] = []
        summary_frames: list[pd.DataFrame] = []
        for baseline_name, hard_name in comparisons:
            per_query, summary = native_model_ranking_jaccard_tables(
                artifacts[baseline_name],
                artifacts[hard_name],
                ground_truth,
                coverage_frame,
                bootstrap_replicates=args.bootstrap_replicates,
                seed=args.seed,
            )
            comparison_name = _safe_variant_name(f"{baseline_name}__vs__{hard_name}")
            comparison_dir = output_dir / "shared_pool_comparisons" / comparison_name
            comparison_dir.mkdir(parents=True, exist_ok=True)
            per_query.to_csv(
                comparison_dir / "native_ranking_jaccard_by_query.csv", index=False
            )
            summary.to_csv(
                comparison_dir / "native_ranking_jaccard_summary.csv", index=False
            )
            query_frames.append(per_query)
            summary_frames.append(summary)
        pd.concat(query_frames, ignore_index=True).to_csv(
            output_dir / "native_ranking_jaccard_by_query.csv", index=False
        )
        pd.concat(summary_frames, ignore_index=True).to_csv(
            output_dir / "native_ranking_jaccard_summary.csv", index=False
        )
        pd.concat(summary_frames, ignore_index=True).to_csv(
            output_dir / "native_topk_retrieval_summary.csv", index=False
        )
        print(f"[done:jaccard] {output_dir}", flush=True)
        return

    if args.cross_model_rerank_only:
        if not comparisons:
            raise ValueError(
                "--cross-model-rerank-only requires at least one --comparison"
            )
        summary_frames: list[pd.DataFrame] = []
        query_frames: list[pd.DataFrame] = []
        for baseline_name, hard_name in comparisons:
            for retriever_name, reranker_name in (
                (baseline_name, hard_name),
                (hard_name, baseline_name),
            ):
                summary, queries = cross_model_positive_query_rerank(
                    artifacts[retriever_name],
                    artifacts[reranker_name],
                    ground_truth,
                    coverage_frame,
                    output_dir=output_dir,
                    pinpoint_root=args.pinpoint_root.resolve(),
                    ground_truth_path=args.ground_truth.resolve(),
                    depth=args.top_k,
                    bootstrap_replicates=args.bootstrap_replicates,
                    seed=args.seed,
                )
                summary_frames.append(summary)
                query_frames.append(queries)
        pd.concat(summary_frames, ignore_index=True).to_csv(
            output_dir / "cross_model_rerank_results.csv", index=False
        )
        pd.concat(query_frames, ignore_index=True).to_parquet(
            output_dir / "cross_model_rerank_by_query.parquet",
            index=False,
            compression="zstd",
        )
        print(f"[done:cross-rerank] {output_dir}", flush=True)
        return

    candidate_summaries: list[dict[str, Any]] = []
    text_sensitivity_frames: list[pd.DataFrame] = []
    dimension_frames: list[pd.DataFrame] = []
    probe_frames: list[pd.DataFrame] = []
    paired_feature_frames: list[pd.DataFrame] = []
    rerank_frames: list[pd.DataFrame] = []
    rank_summary_frames: list[pd.DataFrame] = []
    negative_embeddings_by_model: dict[str, np.ndarray] = {}
    rerank_metadata: dict[str, Any] = {}
    rerank_query_metrics_by_model: dict[str, dict[str, pd.DataFrame]] = {}
    cross_model_rerank_frames: list[pd.DataFrame] = []
    cross_model_rerank_query_frames: list[pd.DataFrame] = []

    for spec in model_specs:
        artifact = artifacts[spec.name]
        print(f"[model] {spec.name}: strict negative query encoding", flush=True)
        negative_embeddings = encode_negative_queries(
            artifact,
            negatives,
            output_dir=output_dir,
            image_cache=args.pinpoint_root.resolve() / "image_cache",
            device=args.device,
            dtype=args.dtype,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_batches=args.prefetch_batches,
            processor_workers=args.processor_workers,
            processor_prefetch_batches=args.processor_prefetch_batches,
            flush_interval=args.flush_interval,
            force=args.force_negative_queries,
            analysis_only=args.analysis_only,
        )
        negative_embeddings_by_model[spec.name] = negative_embeddings
        print(f"[model] {spec.name}: native candidate profiles", flush=True)
        profile, candidate_summary = build_candidate_profile_frame(
            artifact,
            ground_truth,
            groups,
            positive_queries,
            negatives,
            negative_embeddings,
            coverage_frame,
            validated_positive_ids,
            pool_name="native_top50",
            candidate_pools=None,
            seed=args.seed,
        )
        candidate_summaries.append(candidate_summary)
        model_dir = output_dir / spec.name
        model_dir.mkdir(parents=True, exist_ok=True)
        text_pairs, text_summary = negative_text_sensitivity_tables(
            artifact,
            ground_truth,
            negatives,
            negative_embeddings,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
        )
        text_pairs.to_csv(
            model_dir / "negative_text_sensitivity_by_edit.csv", index=False
        )
        text_summary.to_csv(
            model_dir / "negative_text_sensitivity_summary.csv", index=False
        )
        text_sensitivity_frames.append(text_summary)
        profile.to_parquet(
            model_dir / "candidate_profiles_native.parquet",
            index=False,
            compression="zstd",
        )
        write_json_atomic(
            model_dir / "candidate_profile_summary.json", candidate_summary
        )

        depth_dimension_stats = []
        for candidate_depth in (10, 20, 50):
            depth_profile = profile.copy()
            depth_profile["in_pool"] = depth_profile["native_rank"].le(candidate_depth)
            depth_profile["pool_name"] = f"native_top{candidate_depth}"
            stats_at_depth, _ = candidate_dimension_statistics(
                depth_profile,
                bootstrap_replicates=args.bootstrap_replicates,
                seed=args.seed + candidate_depth,
            )
            stats_at_depth["candidate_depth"] = candidate_depth
            depth_dimension_stats.append(stats_at_depth)
            del depth_profile
        dimension_stats = pd.concat(depth_dimension_stats, ignore_index=True)
        dimension_stats.to_csv(model_dir / "profile_dimension_stats.csv", index=False)
        dimension_frames.append(dimension_stats)
        rank_frame = rank_violation_frame(profile)
        rank_frame.to_csv(model_dir / "rank_violations_by_query.csv", index=False)
        rank_summary = summarize_rank_violations(
            rank_frame,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
        )
        rank_summary.to_csv(model_dir / "rank_violation_summary.csv", index=False)
        rank_summary_frames.append(rank_summary)

        primary_pool_predictions = pd.DataFrame()
        if not args.skip_probes:
            print(f"[model] {spec.name}: grouped OOF probes", flush=True)
            (
                probe_results,
                clean_predictions,
                primary_pool_predictions,
                paired_feature_results,
            ) = run_all_probes(
                profile,
                fold_map=fold_map,
                bootstrap_replicates=args.bootstrap_replicates,
                seed=args.seed,
            )
            probe_results.to_csv(model_dir / "probe_results.csv", index=False)
            paired_feature_results.to_csv(
                model_dir / "paired_feature_probe_differences.csv", index=False
            )
            clean_predictions.to_parquet(
                model_dir / "probe_clean_oof_predictions.parquet",
                index=False,
                compression="zstd",
            )
            primary_pool_predictions.to_parquet(
                model_dir / "primary_rerank_oof_predictions.parquet",
                index=False,
                compression="zstd",
            )
            probe_frames.append(probe_results)
            paired_feature_frames.append(paired_feature_results)

        if not args.skip_rerank:
            print(f"[model] {spec.name}: full official rerank evaluation", flush=True)
            (
                rerank_results,
                model_rerank_metadata,
                model_rerank_query_metrics,
            ) = run_rerank_experiments(
                artifact,
                profile,
                primary_pool_predictions,
                ground_truth,
                coverage_frame,
                output_dir=output_dir,
                pinpoint_root=args.pinpoint_root.resolve(),
                ground_truth_path=args.ground_truth.resolve(),
                rerank_depths=rerank_depths,
                bootstrap_replicates=args.bootstrap_replicates,
                seed=args.seed,
                official_eval_all=args.official_eval_all,
            )
            rerank_frames.append(rerank_results)
            rerank_metadata[spec.name] = model_rerank_metadata
            rerank_query_metrics_by_model[spec.name] = model_rerank_query_metrics
        del profile
        gc.collect()

    if not args.skip_rerank:
        for baseline_name, hard_name in comparisons:
            for retriever_name, reranker_name in (
                (baseline_name, hard_name),
                (hard_name, baseline_name),
            ):
                print(
                    f"[cross-rerank] retrieve={retriever_name} rerank={reranker_name}",
                    flush=True,
                )
                cross_summary, cross_queries = cross_model_positive_query_rerank(
                    artifacts[retriever_name],
                    artifacts[reranker_name],
                    ground_truth,
                    coverage_frame,
                    output_dir=output_dir,
                    pinpoint_root=args.pinpoint_root.resolve(),
                    ground_truth_path=args.ground_truth.resolve(),
                    depth=args.top_k,
                    bootstrap_replicates=args.bootstrap_replicates,
                    seed=args.seed,
                )
                cross_model_rerank_frames.append(cross_summary)
                cross_model_rerank_query_frames.append(cross_queries)

    audit_frame = cohort_representativeness_audit(
        ground_truth, coverage_frame, list(artifacts.values())
    )
    audit_frame.to_csv(output_dir / "cohort_representativeness.csv", index=False)
    cohort_composition_frame(ground_truth, coverage_frame).to_csv(
        output_dir / "cohort_composition.csv", index=False
    )

    paired_frames: list[pd.DataFrame] = []
    shared_dimension_frames: list[pd.DataFrame] = []
    shared_probe_frames: list[pd.DataFrame] = []
    shared_feature_pair_frames: list[pd.DataFrame] = []
    shared_model_probe_pair_frames: list[pd.DataFrame] = []
    shared_incremental_probe_did_frames: list[pd.DataFrame] = []
    native_jaccard_query_frames: list[pd.DataFrame] = []
    native_jaccard_summary_frames: list[pd.DataFrame] = []
    rerank_delta_of_difference_frames: list[pd.DataFrame] = []
    shared_pool_metadata: list[dict[str, Any]] = []
    for baseline_name, hard_name in comparisons:
        print(
            f"[shared-pool] {baseline_name} vs {hard_name}",
            flush=True,
        )
        baseline_artifact = artifacts[baseline_name]
        hard_artifact = artifacts[hard_name]
        shared_pools, fingerprint = build_shared_union_pool(
            baseline_artifact, hard_artifact, sorted(pilot_query_ids)
        )
        comparison_name = _safe_variant_name(f"{baseline_name}__vs__{hard_name}")
        comparison_dir = output_dir / "shared_pool_comparisons" / comparison_name
        comparison_dir.mkdir(parents=True, exist_ok=True)
        jaccard_by_query, jaccard_summary = native_model_ranking_jaccard_tables(
            baseline_artifact,
            hard_artifact,
            ground_truth,
            coverage_frame,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
        )
        jaccard_by_query.to_csv(
            comparison_dir / "native_ranking_jaccard_by_query.csv", index=False
        )
        jaccard_summary.to_csv(
            comparison_dir / "native_ranking_jaccard_summary.csv", index=False
        )
        native_jaccard_query_frames.append(jaccard_by_query)
        native_jaccard_summary_frames.append(jaccard_summary)
        pool_name = f"shared_union::{baseline_name}::{hard_name}"
        values_by_model: dict[str, dict[str, pd.DataFrame]] = {}
        probe_query_metrics_by_model: dict[
            str, dict[tuple[str, str], pd.DataFrame]
        ] = {}
        for model_name in (baseline_name, hard_name):
            shared_profile, shared_summary = build_candidate_profile_frame(
                artifacts[model_name],
                ground_truth,
                groups,
                positive_queries,
                negatives,
                negative_embeddings_by_model[model_name],
                coverage_frame,
                validated_positive_ids,
                pool_name=pool_name,
                candidate_pools=shared_pools,
                seed=args.seed,
            )
            candidate_summaries.append(shared_summary)
            if args.save_shared_profiles:
                shared_profile.to_parquet(
                    comparison_dir
                    / f"{_safe_variant_name(model_name)}_candidate_profiles.parquet",
                    index=False,
                    compression="zstd",
                )
            stats, query_values = candidate_dimension_statistics(
                shared_profile,
                bootstrap_replicates=args.bootstrap_replicates,
                seed=args.seed,
            )
            stats.to_csv(
                comparison_dir
                / f"{_safe_variant_name(model_name)}_dimension_stats.csv",
                index=False,
            )
            shared_dimension_frames.append(stats)
            values_by_model[model_name] = query_values
            if not args.skip_probes:
                print(
                    f"[shared-pool] {model_name}: grouped OOF probes",
                    flush=True,
                )
                (
                    shared_probe_results,
                    shared_clean_predictions,
                    _,
                    shared_feature_pairs,
                ) = run_all_probes(
                    shared_profile,
                    fold_map=fold_map,
                    bootstrap_replicates=args.bootstrap_replicates,
                    seed=args.seed,
                    retain_primary_pool_predictions=False,
                )
                safe_model_name = _safe_variant_name(model_name)
                shared_probe_results.to_csv(
                    comparison_dir / f"{safe_model_name}_probe_results.csv",
                    index=False,
                )
                shared_feature_pairs.to_csv(
                    comparison_dir
                    / f"{safe_model_name}_paired_feature_probe_differences.csv",
                    index=False,
                )
                shared_probe_frames.append(shared_probe_results)
                shared_feature_pair_frames.append(shared_feature_pairs)
                probe_query_metrics_by_model[model_name] = probe_query_metrics_by_key(
                    shared_clean_predictions
                )
                del shared_clean_predictions
            del shared_profile
            gc.collect()
        paired = paired_model_profile_statistics(
            values_by_model[baseline_name],
            values_by_model[hard_name],
            baseline_name=baseline_name,
            hard_name=hard_name,
            pool_name=pool_name,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
        )
        paired.to_csv(comparison_dir / "paired_model_differences.csv", index=False)
        paired_frames.append(paired)
        if not args.skip_probes:
            paired_probe = paired_model_probe_statistics(
                probe_query_metrics_by_model[baseline_name],
                probe_query_metrics_by_model[hard_name],
                baseline_name=baseline_name,
                hard_name=hard_name,
                pool_name=pool_name,
                bootstrap_replicates=args.bootstrap_replicates,
                seed=args.seed,
            )
            paired_probe.to_csv(
                comparison_dir / "paired_model_probe_differences.csv", index=False
            )
            shared_model_probe_pair_frames.append(paired_probe)
            incremental_probe_did = paired_model_incremental_probe_did(
                probe_query_metrics_by_model[baseline_name],
                probe_query_metrics_by_model[hard_name],
                baseline_name=baseline_name,
                hard_name=hard_name,
                pool_name=pool_name,
                bootstrap_replicates=args.bootstrap_replicates,
                seed=args.seed,
            )
            incremental_probe_did.to_csv(
                comparison_dir / "incremental_feature_probe_did.csv", index=False
            )
            shared_incremental_probe_did_frames.append(incremental_probe_did)
        if (
            baseline_name in rerank_query_metrics_by_model
            and hard_name in rerank_query_metrics_by_model
        ):
            rerank_delta_of_differences = paired_rerank_delta_of_differences(
                rerank_query_metrics_by_model[baseline_name],
                rerank_query_metrics_by_model[hard_name],
                baseline_name=baseline_name,
                hard_name=hard_name,
                coverage_frame=coverage_frame,
                bootstrap_replicates=args.bootstrap_replicates,
                seed=args.seed,
            )
            rerank_delta_of_differences.to_csv(
                comparison_dir / "rerank_delta_of_differences.csv", index=False
            )
            rerank_delta_of_difference_frames.append(rerank_delta_of_differences)
        pool_metadata = {
            "baseline_model": baseline_name,
            "hard_model": hard_name,
            "pool_name": pool_name,
            "pool_fingerprint": fingerprint,
            "query_count": len(shared_pools),
            "mean_candidate_count": float(
                np.mean([len(values) for values in shared_pools.values()])
            ),
            "max_candidate_count": max(len(values) for values in shared_pools.values()),
        }
        shared_pool_metadata.append(pool_metadata)
        write_json_atomic(comparison_dir / "shared_pool_metadata.json", pool_metadata)

    combined_dimensions = (
        pd.concat(dimension_frames, ignore_index=True)
        if dimension_frames
        else pd.DataFrame()
    )
    combined_text_sensitivity = (
        pd.concat(text_sensitivity_frames, ignore_index=True)
        if text_sensitivity_frames
        else pd.DataFrame()
    )
    combined_probes = (
        pd.concat(probe_frames, ignore_index=True) if probe_frames else pd.DataFrame()
    )
    combined_paired_features = (
        pd.concat(paired_feature_frames, ignore_index=True)
        if paired_feature_frames
        else pd.DataFrame()
    )
    combined_reranks = (
        pd.concat(rerank_frames, ignore_index=True) if rerank_frames else pd.DataFrame()
    )
    combined_rank_summaries = (
        pd.concat(rank_summary_frames, ignore_index=True)
        if rank_summary_frames
        else pd.DataFrame()
    )
    combined_paired = (
        pd.concat(paired_frames, ignore_index=True) if paired_frames else pd.DataFrame()
    )
    combined_shared_dimensions = (
        pd.concat(shared_dimension_frames, ignore_index=True)
        if shared_dimension_frames
        else pd.DataFrame()
    )
    combined_shared_probes = (
        pd.concat(shared_probe_frames, ignore_index=True)
        if shared_probe_frames
        else pd.DataFrame()
    )
    combined_shared_feature_pairs = (
        pd.concat(shared_feature_pair_frames, ignore_index=True)
        if shared_feature_pair_frames
        else pd.DataFrame()
    )
    combined_shared_model_probe_pairs = (
        pd.concat(shared_model_probe_pair_frames, ignore_index=True)
        if shared_model_probe_pair_frames
        else pd.DataFrame()
    )
    combined_shared_incremental_probe_did = (
        pd.concat(shared_incremental_probe_did_frames, ignore_index=True)
        if shared_incremental_probe_did_frames
        else pd.DataFrame()
    )
    combined_rerank_delta_of_differences = (
        pd.concat(rerank_delta_of_difference_frames, ignore_index=True)
        if rerank_delta_of_difference_frames
        else pd.DataFrame()
    )
    combined_cross_model_reranks = (
        pd.concat(cross_model_rerank_frames, ignore_index=True)
        if cross_model_rerank_frames
        else pd.DataFrame()
    )
    combined_cross_model_rerank_queries = (
        pd.concat(cross_model_rerank_query_frames, ignore_index=True)
        if cross_model_rerank_query_frames
        else pd.DataFrame()
    )
    combined_native_jaccard_by_query = (
        pd.concat(native_jaccard_query_frames, ignore_index=True)
        if native_jaccard_query_frames
        else pd.DataFrame()
    )
    combined_native_jaccard_summary = (
        pd.concat(native_jaccard_summary_frames, ignore_index=True)
        if native_jaccard_summary_frames
        else pd.DataFrame()
    )
    decision_summary = build_model_decision_summary(
        combined_text_sensitivity,
        combined_probes,
        combined_paired_features,
        combined_reranks,
        primary_depth=max(rerank_depths),
    )
    primary_did_mask = (
        combined_shared_incremental_probe_did.get("scope", pd.Series(dtype=object)).eq(
            PRIMARY_PROBE_SCOPE
        )
        & combined_shared_incremental_probe_did.get(
            "enhanced_feature", pd.Series(dtype=object)
        ).eq(PRIMARY_PROBE_FEATURE)
        & combined_shared_incremental_probe_did.get(
            "control_feature", pd.Series(dtype=object)
        ).eq("source_similarity_strict")
        & combined_shared_incremental_probe_did.get(
            "metric", pd.Series(dtype=object)
        ).eq("query_pairwise_auc")
    )
    combined_shared_incremental_probe_did["is_primary"] = primary_did_mask
    pair_decision_summary = combined_shared_incremental_probe_did.loc[
        primary_did_mask
    ].copy()
    target_exposure_audit = combined_dimensions.loc[
        combined_dimensions.get("contrast", pd.Series(dtype=object)).eq(
            "validated_minus_unseen_positive"
        )
        & combined_dimensions.get("candidate_depth", pd.Series(dtype=float)).eq(50)
    ].copy()
    combined_dimensions.to_csv(
        output_dir / "profile_dimension_stats_all_models.csv", index=False
    )
    target_exposure_audit.to_csv(
        output_dir / "target_exposure_audit_all_models.csv", index=False
    )
    combined_text_sensitivity.to_csv(
        output_dir / "negative_text_sensitivity_all_models.csv", index=False
    )
    combined_probes.to_csv(output_dir / "probe_results_all_models.csv", index=False)
    combined_paired_features.to_csv(
        output_dir / "paired_feature_probe_differences_all_models.csv", index=False
    )
    combined_reranks.to_csv(output_dir / "rerank_results_all_models.csv", index=False)
    combined_rank_summaries.to_csv(
        output_dir / "rank_violation_summary_all_models.csv", index=False
    )
    combined_paired.to_csv(
        output_dir / "shared_pool_paired_differences.csv", index=False
    )
    combined_shared_dimensions.to_csv(
        output_dir / "shared_pool_dimension_stats_all_models.csv", index=False
    )
    combined_shared_probes.to_csv(
        output_dir / "shared_pool_probe_results_all_models.csv", index=False
    )
    combined_shared_feature_pairs.to_csv(
        output_dir / "shared_pool_paired_feature_differences_all_models.csv",
        index=False,
    )
    combined_shared_model_probe_pairs.to_csv(
        output_dir / "shared_pool_paired_model_probe_differences.csv", index=False
    )
    combined_shared_incremental_probe_did.to_csv(
        output_dir / "shared_pool_incremental_feature_did.csv", index=False
    )
    combined_rerank_delta_of_differences.to_csv(
        output_dir / "rerank_delta_of_differences.csv", index=False
    )
    combined_native_jaccard_by_query.to_csv(
        output_dir / "native_ranking_jaccard_by_query.csv", index=False
    )
    combined_native_jaccard_summary.to_csv(
        output_dir / "native_ranking_jaccard_summary.csv", index=False
    )
    combined_native_jaccard_summary.to_csv(
        output_dir / "native_topk_retrieval_summary.csv", index=False
    )
    combined_cross_model_reranks.to_csv(
        output_dir / "cross_model_rerank_results.csv", index=False
    )
    if not combined_cross_model_rerank_queries.empty:
        combined_cross_model_rerank_queries.to_parquet(
            output_dir / "cross_model_rerank_by_query.parquet",
            index=False,
            compression="zstd",
        )
    decision_summary.to_csv(output_dir / "MODEL_DECISION_SUMMARY.csv", index=False)
    pair_decision_summary.to_csv(output_dir / "PAIR_DECISION_SUMMARY.csv", index=False)
    pd.DataFrame(candidate_summaries).to_csv(
        output_dir / "candidate_profile_summaries.csv", index=False
    )

    metadata = {
        "protocol_version": 1,
        "input": str(args.input.resolve()),
        "ground_truth": str(args.ground_truth.resolve()),
        "models": model_names,
        "model_artifacts": [
            {
                "name": spec.name,
                "checkpoint": str(spec.checkpoint),
                "official_cache": str(spec.official_cache),
                "corpus_cache": str(
                    spec.corpus_cache or spec.official_cache / "cache" / "corpus"
                ),
                "shared_frozen_corpus": spec.allow_shared_frozen_corpus,
            }
            for spec in model_specs
        ],
        "comparisons": comparisons,
        "loader_coverage": loader_coverage,
        "strict_negative_count": len(negatives),
        "coverage": coverage_summary,
        "fold_count": args.folds,
        "fold_assignment_fingerprint": stable_hash(fold_map),
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "rerank_depths": rerank_depths,
        "primary_probe_feature": PRIMARY_PROBE_FEATURE,
        "primary_probe_scope": PRIMARY_PROBE_SCOPE,
        "analysis_mode": (
            "simple_no_ml" if args.skip_probes else "simple_plus_oof_probe"
        ),
        "oof_probes_enabled": not args.skip_probes,
        "official_full_query_count": len(ground_truth),
        "pilot_query_count": len(pilot_query_ids),
        "candidate_summaries": candidate_summaries,
        "shared_pools": shared_pool_metadata,
        "rerank": rerank_metadata,
        "interpretation": {
            "headline_benchmark": "official full PinPoint",
            "response_profile_subset": "V7 500-group official-query subset",
            "missing_edit_cells": "NaN with masks; never zero-filled in profile data",
            "response_profile_status": (
                "target-informed diagnostic only; V7 q- saw positive targets"
            ),
            "cross_model_rerank_status": (
                "no negative text and no learned probe; q+ similarity only"
            ),
        },
    }
    write_json_atomic(output_dir / "run_metadata.json", metadata)
    write_analysis_report(
        output_dir / "COUNTERFACTUAL_PROFILE_ANALYSIS.md",
        coverage_summary=coverage_summary,
        candidate_summaries=candidate_summaries,
        text_sensitivity_results=combined_text_sensitivity,
        target_exposure_audit=target_exposure_audit,
        decision_summary=decision_summary,
        pair_decision_summary=pair_decision_summary,
        probe_results=combined_probes,
        paired_feature_results=combined_paired_features,
        rerank_results=combined_reranks,
        cross_model_rerank_results=combined_cross_model_reranks,
        paired_results=combined_paired,
        native_jaccard_summary=combined_native_jaccard_summary,
        shared_probe_pair_results=combined_shared_model_probe_pairs,
        shared_incremental_probe_did=combined_shared_incremental_probe_did,
        rerank_delta_of_differences=combined_rerank_delta_of_differences,
    )
    print(f"[done] {output_dir}", flush=True)


if __name__ == "__main__":
    main()
