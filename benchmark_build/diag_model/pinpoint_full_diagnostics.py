#!/usr/bin/env python3
"""Full-corpus diagnostics for PinPoint V7 positive and generated-negative text.

The expensive parts are cached per model:

1. native candidate-image embeddings for the complete PinPoint corpus;
2. composed-query embeddings for every original instruction and accepted V7
   hard-negative instruction;
3. exact full-corpus Top-K results.

Offline analysis then reports:

- raw and native-paraphrase-calibrated target win rates;
- NDCG for positive queries and residual-positive NDCG for negative queries;
- Top-K retrieval-set Jaccard for positive paraphrases and positive/negative
  query pairs.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import math
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from common import read_jsonl, sha256_file, write_json_atomic
from trained_clip_adapter import TRAINED_CLIP_MODELS, TrainedCLIPAdapter


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]
CIR_PLAY_ROOT = WORKSPACE_ROOT / "CIR_PLAY"
PINPOINT_ROOT = WORKSPACE_ROOT / "pinpoint-dataset"
CHECKPOINT_ROOT = CIR_PLAY_ROOT / "models" / "checkpoints"
MODEL_PATHS = {
    "BGE-VL-base": CHECKPOINT_ROOT / "BGE-VL-base",
    "BGE-VL-large-original": CHECKPOINT_ROOT / "BGE-VL-large",
    "BGE-VL-MLLM-S1": CHECKPOINT_ROOT / "BGE-VL-MLLM-S1",
    **TRAINED_CLIP_MODELS,
}
DEFAULT_BATCH_SIZES = {
    "BGE-VL-base": 1024,
    "BGE-VL-large-original": 64,
    "BGE-VL-MLLM-S1": 16,
    "CLIP-L-baseline": 128,
    "CLIP-L-hard-w0.5": 128,
    "CLIP-L-last4-baseline": 64,
    "CLIP-L-last4-hard-w0.5": 64,
    "CLIP-L-full-baseline": 64,
    "CLIP-L-full-hard-w0.1": 64,
    "CLIP-L-full-hard-w0.25": 64,
    "CLIP-L-full-hard-w0.5": 64,
    "CLIP-L-full-hard-w0.75": 64,
    "CLIP-L-full-hard-w1.0": 64,
    "CLIP-L-lora-r16-baseline": 64,
    "CLIP-L-lora-r16-hard-w0.5": 64,
    "BGE-VL-large-full-hard-w0.1": 64,
}
DEFAULT_KS = (1, 5, 10, 50)


@dataclass(frozen=True)
class QuerySpec:
    query_index: int
    query_key: str
    query_kind: str
    group_id: str
    instruction_id: str
    source_instruction_id: str
    negative_id: str
    text: str
    reference_ids: tuple[str, ...]
    semantic_target: str
    edit_operation: str
    query_category: str
    length_category: str
    is_question: bool
    is_imperative: bool


@dataclass(frozen=True)
class GroupSpec:
    group_id: str
    reference_ids: tuple[str, ...]
    positive_ids: tuple[str, ...]
    official_negative_ids: tuple[str, ...]
    instruction_ids: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run metric-only full-corpus PinPoint V7 diagnostics."
    )
    parser.add_argument("--input", required=True, type=Path, help="PinPoint V7 final.jsonl")
    parser.add_argument("--model-name", required=True, choices=tuple(MODEL_PATHS))
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--corpus-cache-dir",
        type=Path,
        default=None,
        help=(
            "Optional reusable candidate-image cache. The two supplied CLIP-L "
            "checkpoints share it because their image backbone is frozen."
        ),
    )
    parser.add_argument("--pinpoint-root", type=Path, default=PINPOINT_ROOT)
    parser.add_argument("--image-cache", type=Path, default=PINPOINT_ROOT / "image_cache")
    parser.add_argument(
        "--index-signatures",
        type=Path,
        default=PINPOINT_ROOT / "index_signatures.txt",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--search-device", default=None)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=min(32, os.cpu_count() or 8))
    parser.add_argument("--prefetch-batches", type=int, default=4)
    parser.add_argument("--processor-workers", type=int, default=2)
    parser.add_argument("--processor-prefetch-batches", type=int, default=2)
    parser.add_argument("--flush-interval", type=int, default=16)
    parser.add_argument("--query-search-batch-size", type=int, default=256)
    parser.add_argument("--corpus-search-chunk-size", type=int, default=8192)
    parser.add_argument(
        "--ks",
        default=",".join(str(value) for value in DEFAULT_KS),
        help="Comma-separated full-corpus cutoffs.",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-groups", type=int, default=None)
    parser.add_argument("--limit-index", type=int, default=None)
    parser.add_argument("--force-corpus", action="store_true")
    parser.add_argument("--force-queries", action="store_true")
    parser.add_argument("--force-search", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    return parser.parse_args()


def _stable_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resolve_image_path(signature: str, image_cache: Path) -> Path | None:
    for suffix in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = image_cache / signature[:2] / f"{signature}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def read_available_signatures(
    path: Path,
    image_cache: Path,
    *,
    limit: int | None,
    report_path: Path,
) -> list[str]:
    signatures = [line.strip() for line in path.open(encoding="utf-8") if line.strip()]
    if limit is not None:
        signatures = signatures[:limit]
    if len(signatures) != len(set(signatures)):
        raise ValueError(f"Duplicate image signatures in {path}")
    available: list[str] = []
    missing: list[str] = []
    for signature in signatures:
        if _resolve_image_path(signature, image_cache) is None:
            missing.append(signature)
        else:
            available.append(signature)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "".join(f"{value}\n" for value in missing),
        encoding="utf-8",
    )
    print(
        f"[corpus] available={len(available)}/{len(signatures)} "
        f"missing={len(missing)}"
    )
    return available


def load_v7_dataset(
    path: Path,
    *,
    limit_groups: int | None,
) -> tuple[list[GroupSpec], list[QuerySpec], dict[str, Any]]:
    groups: list[GroupSpec] = []
    queries: list[QuerySpec] = []
    seen_groups: set[str] = set()
    seen_instruction_ids: set[str] = set()
    seen_negative_ids: set[str] = set()
    group_records = list(read_jsonl(path))
    if limit_groups is not None:
        group_records = group_records[:limit_groups]

    for record in group_records:
        if str(record.get("schema_version")) != "7.0":
            raise ValueError(
                f"Expected PinPoint V7 schema, found {record.get('schema_version')!r}"
            )
        group_id = str(record["group_id"])
        if group_id in seen_groups:
            raise ValueError(f"Duplicate group_id: {group_id}")
        seen_groups.add(group_id)
        reference_ids = tuple(str(item["image_id"]) for item in record["references"])
        if len(reference_ids) not in {1, 2}:
            raise ValueError(f"{group_id} has {len(reference_ids)} references")
        positive_ids = tuple(dict.fromkeys(str(item["image_id"]) for item in record["positive_images"]))
        official_negative_ids = tuple(
            dict.fromkeys(
                str(value)
                for value in record.get("official_negative_image_ids", [])
                if value
            )
        )
        instructions = record["instructions"]
        instruction_ids = tuple(str(item["instruction_id"]) for item in instructions)
        instruction_map = {str(item["instruction_id"]): item for item in instructions}
        if len(instruction_ids) != len(instruction_map):
            raise ValueError(f"Duplicate instruction_id within {group_id}")
        groups.append(
            GroupSpec(
                group_id=group_id,
                reference_ids=reference_ids,
                positive_ids=positive_ids,
                official_negative_ids=official_negative_ids,
                instruction_ids=instruction_ids,
            )
        )
        for instruction in instructions:
            instruction_id = str(instruction["instruction_id"])
            if instruction_id in seen_instruction_ids:
                raise ValueError(f"Duplicate instruction_id: {instruction_id}")
            seen_instruction_ids.add(instruction_id)
            queries.append(
                QuerySpec(
                    query_index=len(queries),
                    query_key=f"positive::{instruction_id}",
                    query_kind="positive",
                    group_id=group_id,
                    instruction_id=instruction_id,
                    source_instruction_id=instruction_id,
                    negative_id="",
                    text=str(instruction["text"]),
                    reference_ids=reference_ids,
                    semantic_target="",
                    edit_operation="",
                    query_category=str(instruction.get("query_category") or ""),
                    length_category=str(instruction.get("length_category") or ""),
                    is_question=bool(instruction.get("is_question", False)),
                    is_imperative=bool(instruction.get("is_imperative", False)),
                )
            )
        for negative in record.get("negatives", []):
            negative_id = str(negative["negative_id"])
            if negative_id in seen_negative_ids:
                raise ValueError(f"Duplicate negative_id: {negative_id}")
            seen_negative_ids.add(negative_id)
            instruction_id = str(negative["instruction_id"])
            if instruction_id not in instruction_map:
                raise ValueError(
                    f"{negative_id} references unknown instruction {instruction_id}"
                )
            source = instruction_map[instruction_id]
            source_text = str(source["text"])
            if str(negative["source_text"]) != source_text:
                raise ValueError(f"{negative_id} source_text does not match instruction")
            semantic = str(negative["semantic_target"])
            operation = str(negative["edit_operation"])
            if semantic not in {"object", "attribute", "predicate"}:
                raise ValueError(f"Unknown semantic_target in {negative_id}: {semantic}")
            if operation not in {"add", "delete", "substitute"}:
                raise ValueError(f"Unknown edit_operation in {negative_id}: {operation}")
            queries.append(
                QuerySpec(
                    query_index=len(queries),
                    query_key=f"negative::{negative_id}",
                    query_kind="negative",
                    group_id=group_id,
                    instruction_id=instruction_id,
                    source_instruction_id=instruction_id,
                    negative_id=negative_id,
                    text=str(negative["text"]),
                    reference_ids=reference_ids,
                    semantic_target=semantic,
                    edit_operation=operation,
                    query_category=str(source.get("query_category") or ""),
                    length_category=str(source.get("length_category") or ""),
                    is_question=bool(source.get("is_question", False)),
                    is_imperative=bool(source.get("is_imperative", False)),
                )
            )

    coverage = {
        "group_count": len(groups),
        "positive_query_count": sum(q.query_kind == "positive" for q in queries),
        "negative_query_count": sum(q.query_kind == "negative" for q in queries),
        "group_without_negative_count": sum(
            not any(q.query_kind == "negative" and q.group_id == group.group_id for q in queries)
            for group in groups
        ),
        "instruction_without_negative_count": sum(
            not any(
                q.query_kind == "negative"
                and q.group_id == group.group_id
                and q.instruction_id == instruction_id
                for q in queries
            )
            for group in groups
            for instruction_id in group.instruction_ids
        ),
    }
    if not groups or not queries:
        raise ValueError("PinPoint V7 input contains no usable groups/queries")
    return groups, queries, coverage


def make_query_frame(queries: Sequence[QuerySpec]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "query_id": [f"diag_{query.query_index:05d}" for query in queries],
            "query_image_signature": [query.reference_ids[0] for query in queries],
            "query_image_signature2": [
                query.reference_ids[1] if len(query.reference_ids) == 2 else None
                for query in queries
            ],
            "instruction": [query.text for query in queries],
        }
    )


def query_fingerprint(
    queries: Sequence[QuerySpec],
    *,
    model_name: str,
    model_path: Path,
    dtype: str,
) -> str:
    artifact_sha256 = sha256_file(model_path) if model_path.is_file() else None
    return _stable_hash(
        {
            "protocol_version": 2,
            "model_name": model_name,
            "model_path": str(model_path.resolve()),
            "model_artifact_sha256": artifact_sha256,
            "dtype": dtype,
            "queries": [
                {
                    "key": query.query_key,
                    "references": query.reference_ids,
                    "text": query.text,
                }
                for query in queries
            ],
        }
    )


def load_or_build_query_embeddings(
    adapter: Any,
    queries: Sequence[QuerySpec],
    *,
    model_name: str,
    model_path: Path,
    dtype: str,
    image_cache: Path,
    cache_root: Path,
    batch_size: int,
    num_workers: int,
    prefetch_batches: int,
    processor_workers: int,
    processor_prefetch_batches: int,
    flush_interval: int,
    force: bool,
) -> tuple[np.ndarray, str]:
    if str(CIR_PLAY_ROOT) not in sys.path:
        sys.path.insert(0, str(CIR_PLAY_ROOT))
    from run_pinpoint_bge_lincir import iter_prepared_query_batches

    fingerprint = query_fingerprint(
        queries,
        model_name=model_name,
        model_path=model_path,
        dtype=dtype,
    )
    cache_dir = cache_root / "queries" / fingerprint[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = cache_dir / "query_embeddings.npy"
    keys_path = cache_dir / "query_keys.npy"
    metadata_path = cache_dir / "query_metadata.json"
    expected_keys = [query.query_key for query in queries]
    if (
        embeddings_path.is_file()
        and keys_path.is_file()
        and metadata_path.is_file()
        and not force
    ):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        keys = np.load(keys_path, allow_pickle=False).astype(str).tolist()
        if metadata.get("fingerprint") == fingerprint and keys == expected_keys:
            print(f"[query] loaded cache {embeddings_path}")
            return np.load(embeddings_path, mmap_mode="r"), fingerprint

    temporary = cache_dir / "query_embeddings.tmp.npy"
    progress_path = cache_dir / "query_progress.json"
    if force:
        temporary.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)
        embeddings_path.unlink(missing_ok=True)
        keys_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)

    memmap: np.memmap | None = None
    start_index = 0
    dimension: int | None = None
    if temporary.is_file() and progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if (
            progress.get("fingerprint") == fingerprint
            and int(progress.get("total", -1)) == len(queries)
        ):
            start_index = int(progress["next_index"])
            dimension = int(progress["dimension"])
            memmap = np.lib.format.open_memmap(
                temporary,
                mode="r+",
                dtype=np.float32,
                shape=(len(queries), dimension),
            )
            print(f"[query] resuming at {start_index}/{len(queries)}")

    frame = make_query_frame(queries)
    batches = iter_prepared_query_batches(
        adapter,
        frame,
        image_cache,
        start_idx=start_index,
        batch_size=batch_size,
        num_workers=num_workers,
        image_prefetch_batches=prefetch_batches,
        processor_workers=processor_workers,
        processor_prefetch_batches=processor_prefetch_batches,
    )
    flush_interval = max(1, flush_interval)
    for batch_number, (batch_start, batch_end, _, prepared) in enumerate(
        batches,
        start=1,
    ):
        embeddings = np.asarray(
            adapter.encode_prepared_query_batch(prepared, image_cache),
            dtype=np.float32,
        )
        if embeddings.ndim != 2 or embeddings.shape[0] != batch_end - batch_start:
            raise ValueError(
                f"Query encoder returned {embeddings.shape} for "
                f"{batch_end - batch_start} queries"
            )
        if memmap is None:
            dimension = int(embeddings.shape[1])
            memmap = np.lib.format.open_memmap(
                temporary,
                mode="w+",
                dtype=np.float32,
                shape=(len(queries), dimension),
            )
        memmap[batch_start:batch_end] = embeddings
        if batch_number % flush_interval == 0 or batch_end == len(queries):
            memmap.flush()
            write_json_atomic(
                progress_path,
                {
                    "fingerprint": fingerprint,
                    "total": len(queries),
                    "dimension": dimension,
                    "next_index": batch_end,
                },
            )
        print(f"[query] completed={batch_end}/{len(queries)}")

    if memmap is None:
        if embeddings_path.is_file():
            return np.load(embeddings_path, mmap_mode="r"), fingerprint
        raise RuntimeError("No query embeddings were produced")
    memmap.flush()
    del memmap
    temporary.replace(embeddings_path)
    np.save(keys_path, np.asarray(expected_keys))
    write_json_atomic(
        metadata_path,
        {
            "fingerprint": fingerprint,
            "model_name": model_name,
            "model_path": str(model_path.resolve()),
            "dtype": dtype,
            "query_count": len(queries),
            "embedding_dimension": dimension,
        },
    )
    progress_path.unlink(missing_ok=True)
    print(f"[query] saved cache {embeddings_path}")
    return np.load(embeddings_path, mmap_mode="r"), fingerprint


def exact_topk_torch(
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    *,
    top_k: int,
    device: str,
    query_batch_size: int,
    corpus_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"Search device {device!r} requested, but torch.cuda.is_available() is false"
        )
    query_values = torch.from_numpy(
        np.array(query_embeddings, dtype=np.float32, copy=True)
    ).to(torch_device)
    query_count = query_values.shape[0]
    corpus_count = corpus_embeddings.shape[0]
    top_k = min(top_k, corpus_count)
    best_scores = torch.full(
        (query_count, top_k),
        -torch.inf,
        dtype=torch.float32,
        device=torch_device,
    )
    best_indices = torch.full(
        (query_count, top_k),
        -1,
        dtype=torch.int64,
        device=torch_device,
    )
    query_batch_size = max(1, query_batch_size)
    corpus_chunk_size = max(top_k, corpus_chunk_size)
    chunk_count = math.ceil(corpus_count / corpus_chunk_size)
    with torch.inference_mode():
        for chunk_number, corpus_start in enumerate(
            range(0, corpus_count, corpus_chunk_size),
            start=1,
        ):
            corpus_end = min(corpus_start + corpus_chunk_size, corpus_count)
            corpus_values = torch.from_numpy(
                np.array(
                    corpus_embeddings[corpus_start:corpus_end],
                    dtype=np.float32,
                    copy=True,
                )
            ).to(torch_device)
            local_k = min(top_k, corpus_end - corpus_start)
            for query_start in range(0, query_count, query_batch_size):
                query_end = min(query_start + query_batch_size, query_count)
                scores = query_values[query_start:query_end] @ corpus_values.T
                local_scores, local_indices = torch.topk(
                    scores,
                    k=local_k,
                    dim=1,
                    largest=True,
                    sorted=True,
                )
                local_indices += corpus_start
                combined_scores = torch.cat(
                    (best_scores[query_start:query_end], local_scores),
                    dim=1,
                )
                combined_indices = torch.cat(
                    (best_indices[query_start:query_end], local_indices),
                    dim=1,
                )
                selected_scores, selected_positions = torch.topk(
                    combined_scores,
                    k=top_k,
                    dim=1,
                    largest=True,
                    sorted=True,
                )
                best_scores[query_start:query_end] = selected_scores
                best_indices[query_start:query_end] = torch.gather(
                    combined_indices,
                    1,
                    selected_positions,
                )
            del corpus_values
            if torch_device.type == "cuda":
                torch.cuda.empty_cache()
            print(
                f"[search] corpus_chunk={chunk_number}/{chunk_count} "
                f"images={corpus_end}/{corpus_count}"
            )
    return (
        best_scores.cpu().numpy().astype(np.float32, copy=False),
        best_indices.cpu().numpy().astype(np.int64, copy=False),
    )


def load_or_search(
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    *,
    query_fingerprint_value: str,
    corpus_signatures: Sequence[str],
    cache_root: Path,
    top_k: int,
    device: str,
    query_batch_size: int,
    corpus_chunk_size: int,
    force: bool,
) -> tuple[np.ndarray, np.ndarray, str]:
    search_fingerprint = _stable_hash(
        {
            "protocol_version": 1,
            "query_fingerprint": query_fingerprint_value,
            "corpus_signature_hash": _stable_hash(list(corpus_signatures)),
            "top_k": top_k,
            "method": "exact_inner_product_torch",
        }
    )
    cache_dir = cache_root / "search"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"topk_{search_fingerprint[:16]}.npz"
    if path.is_file() and not force:
        result = np.load(path, allow_pickle=False)
        print(f"[search] loaded cache {path}")
        return result["scores"], result["indices"], search_fingerprint
    scores, indices = exact_topk_torch(
        query_embeddings,
        corpus_embeddings,
        top_k=top_k,
        device=device,
        query_batch_size=query_batch_size,
        corpus_chunk_size=corpus_chunk_size,
    )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, scores=scores, indices=indices)
    temporary.replace(path)
    print(f"[search] saved cache {path}")
    return scores, indices, search_fingerprint


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.size == 0 or values.shape != weights.shape:
        raise ValueError("weighted_quantile requires non-empty equal-shaped arrays")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    cutoff = quantile * float(sorted_weights.sum())
    index = min(int(np.searchsorted(cumulative, cutoff, side="left")), values.size - 1)
    return float(sorted_values[index])


def group_balanced_quantile(
    values_by_group: dict[str, np.ndarray],
    quantile: float,
) -> float:
    values: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for group_values in values_by_group.values():
        array = np.asarray(group_values, dtype=np.float64)
        if array.size == 0:
            continue
        values.append(array)
        weights.append(np.full(array.size, 1.0 / array.size, dtype=np.float64))
    if not values:
        raise ValueError("No group values were available for calibration")
    return weighted_quantile(
        np.concatenate(values),
        np.concatenate(weights),
        quantile,
    )


def bootstrap_group_balanced_quantile(
    values_by_group: dict[str, np.ndarray],
    *,
    quantile: float,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    keys = list(values_by_group)
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = rng.integers(0, len(keys), size=len(keys))
        sampled_values: list[np.ndarray] = []
        sampled_weights: list[np.ndarray] = []
        for index in sampled:
            array = np.asarray(values_by_group[keys[int(index)]], dtype=np.float64)
            if array.size == 0:
                continue
            sampled_values.append(array)
            sampled_weights.append(
                np.full(array.size, 1.0 / array.size, dtype=np.float64)
            )
        estimates[replicate] = weighted_quantile(
            np.concatenate(sampled_values),
            np.concatenate(sampled_weights),
            quantile,
        )
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def ndcg_at_k(
    retrieved_indices: Sequence[int],
    relevant_indices: set[int],
    k: int,
) -> float:
    if not relevant_indices or k <= 0:
        return 0.0
    retrieved = list(retrieved_indices)[:k]
    gains = np.asarray(
        [1.0 if int(index) in relevant_indices else 0.0 for index in retrieved],
        dtype=np.float64,
    )
    discounts = 1.0 / np.log2(np.arange(2, len(retrieved) + 2, dtype=np.float64))
    dcg = float(np.sum(gains * discounts))
    ideal_count = min(len(relevant_indices), k)
    ideal_discounts = 1.0 / np.log2(
        np.arange(2, ideal_count + 2, dtype=np.float64)
    )
    idcg = float(np.sum(ideal_discounts))
    return dcg / idcg if idcg > 0.0 else 0.0


def recall_at_k(
    retrieved_indices: Sequence[int],
    relevant_indices: set[int],
    k: int,
) -> float:
    if not relevant_indices:
        return float("nan")
    hits = sum(int(index) in relevant_indices for index in list(retrieved_indices)[:k])
    return float(hits / len(relevant_indices))


def hit_at_k(
    retrieved_indices: Sequence[int],
    relevant_indices: set[int],
    k: int,
) -> float:
    if not relevant_indices:
        return float("nan")
    return float(
        any(int(index) in relevant_indices for index in list(retrieved_indices)[:k])
    )


def topk_jaccard(
    first: Sequence[int],
    second: Sequence[int],
    k: int,
) -> float:
    first_set = set(int(value) for value in list(first)[:k])
    second_set = set(int(value) for value in list(second)[:k])
    union = first_set | second_set
    return float(len(first_set & second_set) / len(union)) if union else 1.0


def _hierarchical_group_values(frame: pd.DataFrame, column: str) -> pd.Series:
    instruction_values = frame.groupby(
        ["group_id", "instruction_id"],
        sort=False,
    )[column].mean()
    return instruction_values.groupby(level="group_id", sort=False).mean()


def _bootstrap_mean(
    values: np.ndarray,
    *,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    indices = rng.integers(0, values.size, size=(replicates, values.size))
    estimates = values[indices].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def summarize_negative_pairs(
    frame: pd.DataFrame,
    *,
    ks: Sequence[int],
    bootstrap_replicates: int,
    seed: int,
) -> pd.DataFrame:
    subset_specs: list[tuple[str, np.ndarray]] = [
        ("all", np.ones(len(frame), dtype=bool)),
        ("strict", (frame["edit_operation"] != "delete").to_numpy()),
        ("add", (frame["edit_operation"] == "add").to_numpy()),
        ("delete", (frame["edit_operation"] == "delete").to_numpy()),
        ("substitute", (frame["edit_operation"] == "substitute").to_numpy()),
    ]
    for semantic in ("object", "attribute", "predicate"):
        for operation in ("add", "delete", "substitute"):
            subset_specs.append(
                (
                    f"{semantic}|{operation}",
                    (
                        (frame["semantic_target"] == semantic)
                        & (frame["edit_operation"] == operation)
                    ).to_numpy(),
                )
            )
    metric_columns = [
        "mean_target_delta",
        "target_raw_win_rate",
        "target_robust_win_rate",
    ]
    for k in ks:
        metric_columns.extend(
            [
                f"positive_recall@{k}",
                f"negative_residual_recall@{k}",
                f"delta_recall@{k}",
                f"recall_win@{k}",
                f"positive_ndcg@{k}",
                f"negative_residual_ndcg@{k}",
                f"delta_ndcg@{k}",
                f"ndcg_win@{k}",
                f"source_negative_top{k}_jaccard",
                f"positive_official_neg_recall@{k}",
                f"negative_official_neg_recall@{k}",
            ]
        )
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for subset_name, mask in subset_specs:
        subset = frame.loc[mask]
        if subset.empty:
            continue
        row: dict[str, Any] = {
            "subset": subset_name,
            "group_count": subset["group_id"].nunique(),
            "instruction_count": subset[["group_id", "instruction_id"]]
            .drop_duplicates()
            .shape[0],
            "negative_count": len(subset),
            "positive_target_comparison_count": int(subset["positive_target_count"].sum()),
        }
        for column in metric_columns:
            group_values = _hierarchical_group_values(subset, column)
            row[f"pair_balanced_{column}"] = float(group_values.mean())
        for column in (
            "target_raw_win_rate",
            "target_robust_win_rate",
            "delta_ndcg@10",
            "source_negative_top10_jaccard",
        ):
            if column not in subset:
                continue
            group_values = _hierarchical_group_values(subset, column)
            low, high = _bootstrap_mean(
                group_values.to_numpy(dtype=np.float64),
                replicates=bootstrap_replicates,
                rng=rng,
            )
            row[f"{column}_ci_low"] = low
            row[f"{column}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def analyze(
    groups: Sequence[GroupSpec],
    queries: Sequence[QuerySpec],
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    corpus_signatures: Sequence[str],
    top_indices: np.ndarray,
    *,
    model_name: str,
    ks: Sequence[int],
    bootstrap_replicates: int,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    signature_index = {
        str(signature): index for index, signature in enumerate(corpus_signatures)
    }
    group_map = {group.group_id: group for group in groups}
    positive_query_index = {
        (query.group_id, query.instruction_id): query.query_index
        for query in queries
        if query.query_kind == "positive"
    }
    positive_indices_by_group: dict[str, list[int]] = {}
    official_negative_indices_by_group: dict[str, set[int]] = {}
    missing_group_positive_ids: dict[str, list[str]] = {}
    overlap_count = 0
    for group in groups:
        available_positive = [
            signature_index[value]
            for value in group.positive_ids
            if value in signature_index
        ]
        missing_positive = [
            value for value in group.positive_ids if value not in signature_index
        ]
        if missing_positive:
            missing_group_positive_ids[group.group_id] = missing_positive
        if not available_positive:
            raise ValueError(f"{group.group_id} has no positive image in the corpus")
        positive_indices_by_group[group.group_id] = available_positive
        positive_set = set(group.positive_ids)
        overlap_count += len(positive_set & set(group.official_negative_ids))
        official_negative_indices_by_group[group.group_id] = {
            signature_index[value]
            for value in group.official_negative_ids
            if value in signature_index and value not in positive_set
        }

    query_ndcg: dict[tuple[int, int], float] = {}
    query_recall: dict[tuple[int, int], float] = {}
    query_hit: dict[tuple[int, int], float] = {}
    query_neg_recall: dict[tuple[int, int], float] = {}
    for query in queries:
        positive_set = set(positive_indices_by_group[query.group_id])
        official_negative_set = official_negative_indices_by_group[query.group_id]
        for k in ks:
            query_recall[(query.query_index, k)] = recall_at_k(
                top_indices[query.query_index],
                positive_set,
                k,
            )
            query_hit[(query.query_index, k)] = hit_at_k(
                top_indices[query.query_index],
                positive_set,
                k,
            )
            query_ndcg[(query.query_index, k)] = ndcg_at_k(
                top_indices[query.query_index],
                positive_set,
                k,
            )
            query_neg_recall[(query.query_index, k)] = recall_at_k(
                top_indices[query.query_index],
                official_negative_set,
                k,
            )

    score_matrices: dict[str, tuple[list[int], np.ndarray]] = {}
    perturbations_by_group: dict[str, np.ndarray] = {}
    for group in groups:
        query_rows = [
            positive_query_index[(group.group_id, instruction_id)]
            for instruction_id in group.instruction_ids
        ]
        target_rows = positive_indices_by_group[group.group_id]
        scores = (
            np.asarray(query_embeddings[query_rows], dtype=np.float32)
            @ np.asarray(corpus_embeddings[target_rows], dtype=np.float32).T
        )
        score_matrices[group.group_id] = (query_rows, scores)
        differences: list[np.ndarray] = []
        for first, second in itertools.combinations(range(len(query_rows)), 2):
            differences.append(np.abs(scores[first] - scores[second]))
        perturbations_by_group[group.group_id] = (
            np.concatenate(differences)
            if differences
            else np.empty(0, dtype=np.float64)
        )

    threshold = group_balanced_quantile(perturbations_by_group, 0.95)
    threshold_ci = bootstrap_group_balanced_quantile(
        perturbations_by_group,
        quantile=0.95,
        replicates=bootstrap_replicates,
        seed=seed,
    )
    calibration = {
        "model_name": model_name,
        "method": "group-balanced 95th percentile of absolute positive-paraphrase score differences",
        "formula": "|sim(f(R,m_i+),t)-sim(f(R,m_j+),t)|",
        "normal_perturbation_threshold": threshold,
        "threshold_bootstrap_ci": list(threshold_ci),
        "calibration_group_count": len(perturbations_by_group),
        "calibration_observation_count": int(
            sum(len(value) for value in perturbations_by_group.values())
        ),
        "group_weighting": "each PinPoint rewrite group has total weight 1",
    }

    positive_rows: list[dict[str, Any]] = []
    for query in queries:
        if query.query_kind != "positive":
            continue
        group = group_map[query.group_id]
        group_query_rows, group_scores = score_matrices[query.group_id]
        score_row = group_query_rows.index(query.query_index)
        row: dict[str, Any] = {
            "group_id": query.group_id,
            "instruction_id": query.instruction_id,
            "text": query.text,
            "reference_count": len(query.reference_ids),
            "positive_image_count": len(positive_indices_by_group[query.group_id]),
            "official_negative_image_count": len(
                official_negative_indices_by_group[query.group_id]
            ),
            "query_category": query.query_category,
            "length_category": query.length_category,
            "is_question": query.is_question,
            "is_imperative": query.is_imperative,
            "mean_positive_similarity": float(group_scores[score_row].mean()),
        }
        for k in ks:
            row[f"recall@{k}"] = query_recall[(query.query_index, k)]
            row[f"hit@{k}"] = query_hit[(query.query_index, k)]
            row[f"ndcg@{k}"] = query_ndcg[(query.query_index, k)]
            row[f"official_neg_recall@{k}"] = query_neg_recall[
                (query.query_index, k)
            ]
        positive_rows.append(row)
    positive_frame = pd.DataFrame(positive_rows)

    negative_rows: list[dict[str, Any]] = []
    query_by_index = {query.query_index: query for query in queries}
    for query in queries:
        if query.query_kind != "negative":
            continue
        source_index = positive_query_index[
            (query.group_id, query.source_instruction_id)
        ]
        positive_target_indices = positive_indices_by_group[query.group_id]
        source_scores = (
            np.asarray(query_embeddings[source_index], dtype=np.float32)
            @ np.asarray(
                corpus_embeddings[positive_target_indices],
                dtype=np.float32,
            ).T
        )
        negative_scores = (
            np.asarray(query_embeddings[query.query_index], dtype=np.float32)
            @ np.asarray(
                corpus_embeddings[positive_target_indices],
                dtype=np.float32,
            ).T
        )
        deltas = source_scores - negative_scores
        source_query = query_by_index[source_index]
        row = {
            "group_id": query.group_id,
            "instruction_id": query.instruction_id,
            "negative_id": query.negative_id,
            "source_text": source_query.text,
            "negative_text": query.text,
            "semantic_target": query.semantic_target,
            "edit_operation": query.edit_operation,
            "reference_count": len(query.reference_ids),
            "positive_target_count": len(positive_target_indices),
            "query_category": query.query_category,
            "length_category": query.length_category,
            "mean_positive_query_target_score": float(source_scores.mean()),
            "mean_negative_query_target_score": float(negative_scores.mean()),
            "mean_target_delta": float(deltas.mean()),
            "median_target_delta": float(np.median(deltas)),
            "min_target_delta": float(deltas.min()),
            "max_target_delta": float(deltas.max()),
            "target_raw_win_rate": float(np.mean(deltas > 0.0)),
            "target_robust_win_rate": float(np.mean(deltas > threshold)),
            "normal_perturbation_threshold": threshold,
        }
        for k in ks:
            positive_recall = query_recall[(source_index, k)]
            negative_recall = query_recall[(query.query_index, k)]
            row[f"positive_recall@{k}"] = positive_recall
            row[f"negative_residual_recall@{k}"] = negative_recall
            row[f"delta_recall@{k}"] = positive_recall - negative_recall
            row[f"recall_win@{k}"] = float(positive_recall > negative_recall)
            positive_ndcg = query_ndcg[(source_index, k)]
            negative_ndcg = query_ndcg[(query.query_index, k)]
            row[f"positive_ndcg@{k}"] = positive_ndcg
            row[f"negative_residual_ndcg@{k}"] = negative_ndcg
            row[f"delta_ndcg@{k}"] = positive_ndcg - negative_ndcg
            row[f"ndcg_win@{k}"] = float(positive_ndcg > negative_ndcg)
            row[f"source_negative_top{k}_jaccard"] = topk_jaccard(
                top_indices[source_index],
                top_indices[query.query_index],
                k,
            )
            row[f"positive_official_neg_recall@{k}"] = query_neg_recall[
                (source_index, k)
            ]
            row[f"negative_official_neg_recall@{k}"] = query_neg_recall[
                (query.query_index, k)
            ]
        negative_rows.append(row)
    negative_frame = pd.DataFrame(negative_rows)

    paraphrase_pair_rows: list[dict[str, Any]] = []
    for group in groups:
        query_rows, score_matrix = score_matrices[group.group_id]
        for first_position, second_position in itertools.combinations(
            range(len(query_rows)),
            2,
        ):
            first_index = query_rows[first_position]
            second_index = query_rows[second_position]
            first_query = query_by_index[first_index]
            second_query = query_by_index[second_index]
            row = {
                "group_id": group.group_id,
                "instruction_id_a": first_query.instruction_id,
                "instruction_id_b": second_query.instruction_id,
                "text_a": first_query.text,
                "text_b": second_query.text,
                "reference_count": len(group.reference_ids),
                "positive_target_count": len(positive_indices_by_group[group.group_id]),
                "mean_absolute_positive_score_difference": float(
                    np.mean(
                        np.abs(
                            score_matrix[first_position]
                            - score_matrix[second_position]
                        )
                    )
                ),
            }
            for k in ks:
                row[f"top{k}_jaccard"] = topk_jaccard(
                    top_indices[first_index],
                    top_indices[second_index],
                    k,
                )
                row[f"absolute_ndcg_difference@{k}"] = abs(
                    query_ndcg[(first_index, k)]
                    - query_ndcg[(second_index, k)]
                )
            paraphrase_pair_rows.append(row)
    paraphrase_pair_frame = pd.DataFrame(paraphrase_pair_rows)

    negative_summary = summarize_negative_pairs(
        negative_frame,
        ks=ks,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    positive_summary: dict[str, Any] = {
        "model_name": model_name,
        "group_count": positive_frame["group_id"].nunique(),
        "positive_query_count": len(positive_frame),
        "positive_paraphrase_pair_count": len(paraphrase_pair_frame),
    }
    for k in ks:
        positive_recall_values = (
            positive_frame.groupby("group_id", sort=False)[f"recall@{k}"].mean()
        )
        positive_hit_values = (
            positive_frame.groupby("group_id", sort=False)[f"hit@{k}"].mean()
        )
        positive_summary[f"pair_balanced_positive_recall@{k}"] = float(
            positive_recall_values.mean()
        )
        positive_summary[f"pair_balanced_positive_hit_rate@{k}"] = float(
            positive_hit_values.mean()
        )
        positive_group_values = (
            positive_frame.groupby("group_id", sort=False)[f"ndcg@{k}"].mean()
        )
        positive_summary[f"pair_balanced_positive_ndcg@{k}"] = float(
            positive_group_values.mean()
        )
        paraphrase_group_values = paraphrase_pair_frame.groupby(
            "group_id",
            sort=False,
        )[f"top{k}_jaccard"].mean()
        positive_summary[
            f"pair_balanced_positive_paraphrase_top{k}_jaccard"
        ] = float(paraphrase_group_values.mean())
        ndcg_ranges = positive_frame.groupby("group_id", sort=False)[
            f"ndcg@{k}"
        ].agg(lambda values: float(values.max() - values.min()))
        positive_summary[f"pair_balanced_positive_ndcg_range@{k}"] = float(
            ndcg_ranges.mean()
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    positive_frame.to_csv(output_dir / "positive_query_metrics.csv", index=False)
    negative_frame.to_csv(output_dir / "negative_pair_metrics.csv", index=False)
    paraphrase_pair_frame.to_csv(
        output_dir / "positive_paraphrase_pair_metrics.csv",
        index=False,
    )
    negative_summary.to_csv(output_dir / "negative_summary.csv", index=False)
    write_json_atomic(output_dir / "calibration.json", calibration)
    write_json_atomic(output_dir / "positive_summary.json", positive_summary)
    return {
        "calibration": calibration,
        "positive_summary": positive_summary,
        "negative_summary": negative_summary,
        "missing_group_positive_ids": missing_group_positive_ids,
        "positive_official_negative_overlap_count": overlap_count,
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates <= 0:
        raise ValueError("--bootstrap-replicates must be positive")
    ks = tuple(
        sorted(
            {
                int(value.strip())
                for value in args.ks.split(",")
                if value.strip()
            }
        )
    )
    if not ks or min(ks) <= 0:
        raise ValueError("--ks must contain positive integers")
    model_path = (args.model_path or MODEL_PATHS[args.model_name]).resolve()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    image_cache = args.image_cache.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    groups, queries, coverage = load_v7_dataset(
        input_path,
        limit_groups=args.limit_groups,
    )
    signatures = read_available_signatures(
        args.index_signatures.resolve(),
        image_cache,
        limit=args.limit_index,
        report_path=output_dir / "missing_corpus_signatures.txt",
    )
    signature_set = set(signatures)
    missing_positive = sorted(
        {
            value
            for group in groups
            for value in group.positive_ids
            if value not in signature_set
        }
    )
    if missing_positive:
        raise ValueError(
            f"{len(missing_positive)} positive images are absent from the searched "
            f"corpus; first={missing_positive[:5]}"
        )
    max_k = min(max(ks), len(signatures))
    cache_root = output_dir / "cache"
    corpus_cache_dir = (
        args.corpus_cache_dir.resolve()
        if args.corpus_cache_dir is not None
        else cache_root / "corpus"
    )
    batch_size = args.batch_size or DEFAULT_BATCH_SIZES[args.model_name]
    requested_search_device = args.search_device or args.device

    if str(CIR_PLAY_ROOT) not in sys.path:
        sys.path.insert(0, str(CIR_PLAY_ROOT))
    from run_pinpoint_bge_lincir import (
        choose_device,
        load_or_build_corpus_embeddings,
        make_adapter,
        prepare_hf_runtime_cache,
    )

    prepare_hf_runtime_cache(cache_root / "_hf_runtime")
    search_device = str(choose_device(requested_search_device))
    model_protocol: dict[str, Any] | None = None
    if args.analysis_only:
        corpus_path = corpus_cache_dir / "corpus_embeddings.npy"
        identifiers_path = corpus_cache_dir / "corpus_signatures.npy"
        if not corpus_path.is_file() or not identifiers_path.is_file():
            raise FileNotFoundError(
                "--analysis-only requested but corpus embedding cache is missing"
            )
        corpus_embeddings = np.load(corpus_path, mmap_mode="r")
        cached_signatures = np.load(identifiers_path, allow_pickle=False).astype(str)
        if cached_signatures.tolist() != signatures:
            raise ValueError("Cached corpus signatures do not match current corpus")
        fingerprint = query_fingerprint(
            queries,
            model_name=args.model_name,
            model_path=model_path,
            dtype=args.dtype,
        )
        query_cache_dir = cache_root / "queries" / fingerprint[:16]
        query_path = query_cache_dir / "query_embeddings.npy"
        if not query_path.is_file():
            raise FileNotFoundError(
                "--analysis-only requested but query embedding cache is missing"
            )
        query_embeddings = np.load(query_path, mmap_mode="r")
        query_fingerprint_value = fingerprint
    else:
        device = choose_device(args.device)
        if args.model_name in TRAINED_CLIP_MODELS:
            adapter = TrainedCLIPAdapter(model_path, device, args.dtype)
            model_protocol = adapter.protocol_metadata()
        else:
            adapter = make_adapter(
                args.model_name,
                model_path,
                device,
                args.dtype,
                hf_cache_dir=CHECKPOINT_ROOT / "lincir" / ".cache" / "huggingface",
                lincir_clip_model_name="large",
            )
        corpus_embeddings, cached_signatures = load_or_build_corpus_embeddings(
            adapter,
            signatures,
            image_cache=image_cache,
            cache_dir=corpus_cache_dir,
            batch_size=batch_size,
            num_workers=args.num_workers,
            prefetch_batches=args.prefetch_batches,
            processor_workers=args.processor_workers,
            processor_prefetch_batches=args.processor_prefetch_batches,
            flush_interval=args.flush_interval,
            force=args.force_corpus,
        )
        query_embeddings, query_fingerprint_value = load_or_build_query_embeddings(
            adapter,
            queries,
            model_name=args.model_name,
            model_path=model_path,
            dtype=args.dtype,
            image_cache=image_cache,
            cache_root=cache_root,
            batch_size=batch_size,
            num_workers=args.num_workers,
            prefetch_batches=args.prefetch_batches,
            processor_workers=args.processor_workers,
            processor_prefetch_batches=args.processor_prefetch_batches,
            flush_interval=args.flush_interval,
            force=args.force_queries,
        )
        del adapter
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    top_scores, top_indices, search_fingerprint = load_or_search(
        query_embeddings,
        corpus_embeddings,
        query_fingerprint_value=query_fingerprint_value,
        corpus_signatures=signatures,
        cache_root=cache_root,
        top_k=max_k,
        device=search_device,
        query_batch_size=args.query_search_batch_size,
        corpus_chunk_size=args.corpus_search_chunk_size,
        force=args.force_search,
    )
    if top_indices.shape != (len(queries), max_k):
        raise ValueError(
            f"Top-K cache shape {top_indices.shape} != {(len(queries), max_k)}"
        )
    del top_scores
    results = analyze(
        groups,
        queries,
        query_embeddings,
        corpus_embeddings,
        signatures,
        top_indices,
        model_name=args.model_name,
        ks=ks,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
        output_dir=output_dir / "analysis",
    )
    metadata = {
        "schema_version": "1.0",
        "model_name": args.model_name,
        "model_path": str(model_path),
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "pinpoint_root": str(args.pinpoint_root.resolve()),
        "image_cache": str(image_cache),
        "corpus_cache_dir": str(corpus_cache_dir),
        "corpus_image_count": len(signatures),
        "query_count": len(queries),
        "ks": list(ks),
        "top_k": max_k,
        "device": args.device,
        "search_device": search_device,
        "dtype": args.dtype,
        "query_fingerprint": query_fingerprint_value,
        "search_fingerprint": search_fingerprint,
        "python": platform.python_version(),
        "coverage": coverage,
        "positive_official_negative_overlap_count": results[
            "positive_official_negative_overlap_count"
        ],
        "missing_group_positive_ids": results["missing_group_positive_ids"],
        "metric_definitions": {
            "recall@k": (
                "number of original positive images retrieved in the full-corpus "
                "Top-K divided by the number of original positive images"
            ),
            "hit@k": "whether at least one original positive image appears in Top-K",
            "negative_residual_ndcg": (
                "NDCG of a generated negative-text query using the original "
                "positive image set as binary relevance; lower is expected"
            ),
            "topk_jaccard": (
                "set Jaccard of two full-corpus Top-K retrieval result sets"
            ),
            "delete": (
                "DELETE is a weaker-query diagnostic and is reported separately "
                "from strict ADD/SUBSTITUTE"
            ),
        },
    }
    if model_protocol is not None:
        metadata["model_protocol"] = model_protocol
    write_json_atomic(output_dir / "metadata.json", metadata)
    print(
        f"[done] model={args.model_name} threshold="
        f"{results['calibration']['normal_perturbation_threshold']:.6f} "
        f"outputs={output_dir / 'analysis'}"
    )


if __name__ == "__main__":
    main()
