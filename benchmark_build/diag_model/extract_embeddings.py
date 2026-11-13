#!/usr/bin/env python3
"""Extract q+, q-, and target embeddings for CIR-QDiag hard negatives.

The expensive model forwards are cached as one NPY file per logical item.  A
completed run assembles those files into features.npz and writes scores.csv.
Supported models use their native composed-query and candidate-image encoders.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from common import (
    batched,
    normalized_rows,
    read_jsonl,
    save_npy_atomic,
    save_npz_atomic,
    sha256_file,
    stable_key,
    validate_label,
    write_json_atomic,
    write_jsonl_atomic,
)
from qwen3vl_adapter import (
    QWEN_MODEL_NAME,
    Qwen3VLDiagnosticAdapter,
    protocol_metadata as qwen_protocol_metadata,
    reference_protocol_metadata as qwen_reference_protocol_metadata,
)


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]
CIR_PLAY_ROOT = WORKSPACE_ROOT / "CIR_PLAY"
DEFAULT_CHECKPOINT_ROOT = CIR_PLAY_ROOT / "models" / "checkpoints"
QWEN_MODEL_ROOT = WORKSPACE_ROOT / "qw3" / QWEN_MODEL_NAME

SUPPORTED_MODELS = {
    "BGE-VL-base": DEFAULT_CHECKPOINT_ROOT / "BGE-VL-base",
    "BGE-VL-MLLM-S1": DEFAULT_CHECKPOINT_ROOT / "BGE-VL-MLLM-S1",
    QWEN_MODEL_NAME: QWEN_MODEL_ROOT,
}

DEFAULT_BATCH_SIZES = {
    "BGE-VL-base": 64,
    "BGE-VL-MLLM-S1": 4,
    QWEN_MODEL_NAME: 4,
}
REFERENCE_BASELINE_PROTOCOL_VERSION = "1"


@dataclass(frozen=True)
class Sample:
    sample_index: int
    sample_id: str
    pair_id: str
    reference_path: str
    target_path: str
    positive_text: str


@dataclass(frozen=True)
class Negative:
    negative_index: int
    sample_index: int
    sample_id: str
    pair_id: str
    negative_id: str
    positive_text: str
    negative_text: str
    semantic_target: str
    edit_operation: str
    construction_mode: str


@dataclass(frozen=True)
class QueryItem:
    cache_key: str
    query_id: str
    image_path: str
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache q+, q-, and t embeddings for CIR-QDiag diagnostics."
    )
    parser.add_argument("--input", required=True, type=Path, help="CIR-QDiag final.jsonl")
    parser.add_argument("--model-name", required=True, choices=tuple(SUPPORTED_MODELS))
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Local checkpoint. Defaults to the registered local path for the model.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, cpu, or auto")
    parser.add_argument("--dtype", default="auto", help="auto, fp16, bf16, or fp32")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Defaults to 64 for BGE-VL-base and 4 for the other models.",
    )
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse compatible per-item NPY files (default: true).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute every required embedding and replace assembled outputs.",
    )
    return parser.parse_args()


def load_dataset(path: Path, limit_samples: int | None) -> tuple[list[Sample], list[Negative]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input JSONL does not exist: {path}")

    samples: list[Sample] = []
    negatives: list[Negative] = []
    seen_sample_ids: set[str] = set()
    seen_negative_ids: set[str] = set()

    for record in read_jsonl(path):
        raw_negatives = record.get("negatives") or []
        if not raw_negatives:
            continue
        if limit_samples is not None and len(samples) >= limit_samples:
            break

        sample_id = str(record["sample_id"])
        if sample_id in seen_sample_ids:
            raise ValueError(f"Duplicate sample_id in input: {sample_id}")
        seen_sample_ids.add(sample_id)

        sample_index = len(samples)
        sample = Sample(
            sample_index=sample_index,
            sample_id=sample_id,
            pair_id=str(record["pair_id"]),
            reference_path=str(record["reference"]["image_path"]),
            target_path=str(record["target"]["image_path"]),
            positive_text=str(record["positive_text"]),
        )
        for image_path in (sample.reference_path, sample.target_path):
            if not Path(image_path).is_file():
                raise FileNotFoundError(f"Image referenced by {sample_id} is missing: {image_path}")
        samples.append(sample)

        for raw_negative in raw_negatives:
            negative_id = str(raw_negative["negative_id"])
            if negative_id in seen_negative_ids:
                raise ValueError(f"Duplicate negative_id in input: {negative_id}")
            seen_negative_ids.add(negative_id)
            semantic_target = str(raw_negative["semantic_target"])
            edit_operation = str(raw_negative["edit_operation"])
            construction_mode = str(raw_negative["construction_mode"])
            validate_label(semantic_target, edit_operation, construction_mode)
            negatives.append(
                Negative(
                    negative_index=len(negatives),
                    sample_index=sample_index,
                    sample_id=sample_id,
                    pair_id=sample.pair_id,
                    negative_id=negative_id,
                    positive_text=sample.positive_text,
                    negative_text=str(raw_negative["text"]),
                    semantic_target=semantic_target,
                    edit_operation=edit_operation,
                    construction_mode=construction_mode,
                )
            )

    if not samples or not negatives:
        raise ValueError("No samples with accepted negatives were found")
    return samples, negatives


def cache_path(cache_root: Path, role: str, logical_key: str) -> Path:
    return cache_root / role / f"{stable_key(logical_key)}.npy"


def _is_cuda_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message and ("cuda" in message or "cublas" in message)


def encode_with_oom_split(
    items: Sequence[Any],
    encode: Callable[[Sequence[Any]], np.ndarray],
) -> np.ndarray:
    try:
        values = np.asarray(encode(items), dtype=np.float32)
        if values.ndim != 2 or values.shape[0] != len(items):
            raise ValueError(
                f"Encoder returned shape {values.shape} for a batch of {len(items)} items"
            )
        return normalized_rows(values)
    except (RuntimeError, MemoryError) as exc:
        if not _is_cuda_oom(exc) or len(items) <= 1:
            raise
        split = len(items) // 2
        print(f"[warn] CUDA OOM; retrying batch as {split}+{len(items) - split}")
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        return np.concatenate(
            (
                encode_with_oom_split(items[:split], encode),
                encode_with_oom_split(items[split:], encode),
            ),
            axis=0,
        )


def make_adapter(args: argparse.Namespace, model_path: Path):
    if str(CIR_PLAY_ROOT) not in sys.path:
        sys.path.insert(0, str(CIR_PLAY_ROOT))
    from run_pinpoint_bge_lincir import (  # type: ignore
        choose_device,
        make_adapter as make_cir_adapter,
        prepare_hf_runtime_cache,
    )

    prepare_hf_runtime_cache(args.output_dir / "hf_runtime")
    device = choose_device(args.device)
    if device.type == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is false. "
                "Run on a GPU node or explicitly pass --device cpu."
            )
    if args.model_name == QWEN_MODEL_NAME:
        return Qwen3VLDiagnosticAdapter(model_path, device, args.dtype)
    return make_cir_adapter(
        args.model_name,
        model_path,
        device,
        args.dtype,
        hf_cache_dir=DEFAULT_CHECKPOINT_ROOT / "lincir" / ".cache" / "huggingface",
        lincir_clip_model_name="large",
    )


def extract_queries(
    adapter: Any,
    items: list[QueryItem],
    cache_root: Path,
    batch_size: int,
    *,
    resume: bool,
    force: bool,
) -> None:
    pending = [
        item
        for item in items
        if force
        or not resume
        or not cache_path(cache_root, "query", item.cache_key).is_file()
    ]
    print(f"[query] cached={len(items) - len(pending)} pending={len(pending)}")
    for batch_number, batch in enumerate(batched(pending, batch_size), start=1):
        def encode(query_items: Sequence[QueryItem]) -> np.ndarray:
            frame = pd.DataFrame(
                {
                    "query_id": [item.query_id for item in query_items],
                    "query_image_signature": [item.image_path for item in query_items],
                    "instruction": [item.text for item in query_items],
                }
            )
            return adapter.encode_query_batch(frame, image_cache=WORKSPACE_ROOT)

        embeddings = encode_with_oom_split(batch, encode)
        for item, embedding in zip(batch, embeddings):
            save_npy_atomic(
                cache_path(cache_root, "query", item.cache_key),
                embedding.astype(np.float32, copy=False),
            )
        print(f"[query] batch={batch_number} completed={min(batch_number * batch_size, len(pending))}/{len(pending)}")


def extract_targets(
    adapter: Any,
    samples: list[Sample],
    cache_root: Path,
    batch_size: int,
    *,
    resume: bool,
    force: bool,
) -> None:
    pending = [
        sample
        for sample in samples
        if force
        or not resume
        or not cache_path(cache_root, "target", sample.sample_id).is_file()
    ]
    print(f"[target] cached={len(samples) - len(pending)} pending={len(pending)}")
    for batch_number, batch in enumerate(batched(pending, batch_size), start=1):

        def encode(targets: Sequence[Sample]) -> np.ndarray:
            return adapter.encode_image_batch([Path(item.target_path) for item in targets])

        embeddings = encode_with_oom_split(batch, encode)
        for sample, embedding in zip(batch, embeddings):
            save_npy_atomic(
                cache_path(cache_root, "target", sample.sample_id),
                embedding.astype(np.float32, copy=False),
            )
        print(f"[target] batch={batch_number} completed={min(batch_number * batch_size, len(pending))}/{len(pending)}")


def encode_reference_query_batch(
    adapter: Any,
    model_name: str,
    image_paths: Sequence[Path],
) -> np.ndarray:
    """Encode f_query(R, empty) without changing the model's query-side role.

    BGE-VL-base composes normalized image and text branches, so removing the
    modification means retaining only its image branch.  The MLLM and Qwen
    variants must keep their query task prompt/instruction; using their
    candidate-image encoder here would confound modality contribution with a
    query/candidate role change.
    """

    if model_name == "BGE-VL-base":
        return adapter.encode_image_batch(image_paths)
    if model_name == "BGE-VL-MLLM-S1":
        return adapter._encode(image_paths, texts=None, q_or_c="q")
    if model_name == QWEN_MODEL_NAME:
        return adapter.encode_reference_query_batch(image_paths)
    raise ValueError(f"Reference-query baseline is not defined for {model_name!r}")


def reference_baseline_protocol(model_name: str) -> dict[str, Any]:
    common = {
        "version": REFERENCE_BASELINE_PROTOCOL_VERSION,
        "role": "query",
        "content": ["reference_image"],
    }
    if model_name == "BGE-VL-base":
        return {**common, "implementation": "normalized_reference_image_branch"}
    if model_name == "BGE-VL-MLLM-S1":
        return {
            **common,
            "implementation": "query_role_with_task_prompt_and_none_modification",
        }
    if model_name == QWEN_MODEL_NAME:
        return {
            **common,
            **qwen_reference_protocol_metadata(),
            "implementation": "image_only_composed_query_instruction",
        }
    raise ValueError(f"Reference-query baseline is not defined for {model_name!r}")


def reference_cache_key(reference_path: str) -> str:
    return f"v{REFERENCE_BASELINE_PROTOCOL_VERSION}:{reference_path}"


def extract_reference_queries(
    adapter: Any,
    model_name: str,
    samples: list[Sample],
    cache_root: Path,
    batch_size: int,
    *,
    resume: bool,
    force: bool,
) -> None:
    pending_by_path: dict[str, Sample] = {}
    for sample in samples:
        path = cache_path(
            cache_root, "reference", reference_cache_key(sample.reference_path)
        )
        if force or not resume or not path.is_file():
            pending_by_path.setdefault(sample.reference_path, sample)
    pending = list(pending_by_path.values())
    unique_reference_count = len(set(sample.reference_path for sample in samples))
    print(
        f"[reference] cached={unique_reference_count - len(pending)} "
        f"pending={len(pending)}"
    )
    for batch_number, batch in enumerate(batched(pending, batch_size), start=1):

        def encode(reference_samples: Sequence[Sample]) -> np.ndarray:
            return encode_reference_query_batch(
                adapter,
                model_name,
                [Path(item.reference_path) for item in reference_samples],
            )

        embeddings = encode_with_oom_split(batch, encode)
        for sample, embedding in zip(batch, embeddings):
            save_npy_atomic(
                cache_path(
                    cache_root,
                    "reference",
                    reference_cache_key(sample.reference_path),
                ),
                embedding.astype(np.float32, copy=False),
            )
        print(
            f"[reference] batch={batch_number} "
            f"completed={min(batch_number * batch_size, len(pending))}/{len(pending)}"
        )


def load_cached_vector(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Expected cached embedding is missing: {path}")
    value = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if value.ndim != 1 or not np.isfinite(value).all():
        raise ValueError(f"Invalid cached vector at {path}: shape={value.shape}")
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError(f"Zero-norm cached vector at {path}")
    return value / norm


def assemble_outputs(
    output_dir: Path,
    cache_root: Path,
    samples: list[Sample],
    negatives: list[Negative],
    run_metadata: dict[str, Any],
) -> dict[str, Any]:
    q_positive = np.stack(
        [
            load_cached_vector(cache_path(cache_root, "query", f"positive:{sample.sample_id}"))
            for sample in samples
        ]
    )
    targets = np.stack(
        [
            load_cached_vector(cache_path(cache_root, "target", sample.sample_id))
            for sample in samples
        ]
    )
    q_reference = np.stack(
        [
            load_cached_vector(
                cache_path(
                    cache_root,
                    "reference",
                    reference_cache_key(sample.reference_path),
                )
            )
            for sample in samples
        ]
    )
    q_negative = np.stack(
        [
            load_cached_vector(cache_path(cache_root, "query", f"negative:{item.negative_id}"))
            for item in negatives
        ]
    )
    dimensions = {
        q_reference.shape[1],
        q_positive.shape[1],
        targets.shape[1],
        q_negative.shape[1],
    }
    if len(dimensions) != 1:
        raise ValueError(f"Embedding dimensions do not agree: {sorted(dimensions)}")

    sample_indices = np.asarray([item.sample_index for item in negatives], dtype=np.int64)
    positive_for_negative = q_positive[sample_indices]
    reference_for_negative = q_reference[sample_indices]
    target_for_negative = targets[sample_indices]
    vectors = positive_for_negative - q_negative
    positive_scores = np.einsum("nd,nd->n", positive_for_negative, target_for_negative)
    negative_scores = np.einsum("nd,nd->n", q_negative, target_for_negative)
    deltas = positive_scores - negative_scores
    delta_from_vector = np.einsum("nd,nd->n", vectors, target_for_negative)
    identity_error = np.abs(deltas - delta_from_vector)
    max_identity_error = float(identity_error.max(initial=0.0))
    if max_identity_error > 1e-5:
        raise AssertionError(
            f"Delta identity failed: max |delta - dot(v,t)| = {max_identity_error:.3e}"
        )

    vector_norms = np.linalg.norm(vectors, axis=1)
    vector_target_cosines = np.divide(
        delta_from_vector,
        vector_norms,
        out=np.zeros_like(delta_from_vector),
        where=vector_norms > 1e-12,
    )
    reference_scores = np.einsum("nd,nd->n", reference_for_negative, target_for_negative)
    positive_gains = positive_scores - reference_scores
    negative_gains = negative_scores - reference_scores
    contribution_identity_error = np.abs(deltas - (positive_gains - negative_gains))
    max_contribution_identity_error = float(
        contribution_identity_error.max(initial=0.0)
    )
    if max_contribution_identity_error > 1e-5:
        raise AssertionError(
            "Reference contribution identity failed: "
            f"max |delta - (G+ - G-)| = {max_contribution_identity_error:.3e}"
        )
    positive_reference_distances = np.linalg.norm(
        positive_for_negative - reference_for_negative, axis=1
    )
    negative_reference_distances = np.linalg.norm(
        q_negative - reference_for_negative, axis=1
    )

    save_npz_atomic(
        output_dir / "features.npz",
        q_reference=q_reference.astype(np.float32, copy=False),
        q_positive=q_positive.astype(np.float32, copy=False),
        target=targets.astype(np.float32, copy=False),
        q_negative=q_negative.astype(np.float32, copy=False),
        sample_index=sample_indices,
    )

    rows = [
        {
            "negative_index": item.negative_index,
            "sample_index": item.sample_index,
            "sample_id": item.sample_id,
            "pair_id": item.pair_id,
            "negative_id": item.negative_id,
            "positive_text": item.positive_text,
            "negative_text": item.negative_text,
            "semantic_target": item.semantic_target,
            "edit_operation": item.edit_operation,
            "construction_mode": item.construction_mode,
        }
        for item in negatives
    ]
    write_jsonl_atomic(output_dir / "rows.jsonl", rows)

    scores_path = output_dir / "scores.csv"
    temporary_scores = scores_path.with_name(f".{scores_path.name}.{os.getpid()}.tmp")
    fieldnames = [
        "negative_index",
        "sample_id",
        "pair_id",
        "negative_id",
        "semantic_target",
        "edit_operation",
        "construction_mode",
        "reference_score",
        "positive_score",
        "negative_score",
        "positive_gain_from_reference",
        "negative_gain_from_reference",
        "delta",
        "positive_reference_distance",
        "negative_reference_distance",
        "vector_norm",
        "vector_target_cosine",
    ]
    with temporary_scores.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for (
            item,
            reference_score,
            positive_score,
            negative_score,
            positive_gain,
            negative_gain,
            delta,
            positive_reference_distance,
            negative_reference_distance,
            vector_norm,
            vector_cosine,
        ) in zip(
            negatives,
            reference_scores,
            positive_scores,
            negative_scores,
            positive_gains,
            negative_gains,
            deltas,
            positive_reference_distances,
            negative_reference_distances,
            vector_norms,
            vector_target_cosines,
        ):
            writer.writerow(
                {
                    "negative_index": item.negative_index,
                    "sample_id": item.sample_id,
                    "pair_id": item.pair_id,
                    "negative_id": item.negative_id,
                    "semantic_target": item.semantic_target,
                    "edit_operation": item.edit_operation,
                    "construction_mode": item.construction_mode,
                    "reference_score": float(reference_score),
                    "positive_score": float(positive_score),
                    "negative_score": float(negative_score),
                    "positive_gain_from_reference": float(positive_gain),
                    "negative_gain_from_reference": float(negative_gain),
                    "delta": float(delta),
                    "positive_reference_distance": float(positive_reference_distance),
                    "negative_reference_distance": float(negative_reference_distance),
                    "vector_norm": float(vector_norm),
                    "vector_target_cosine": float(vector_cosine),
                }
            )
    temporary_scores.replace(scores_path)

    norm_errors = np.concatenate(
        (
            np.abs(np.linalg.norm(q_positive, axis=1) - 1.0),
            np.abs(np.linalg.norm(q_reference, axis=1) - 1.0),
            np.abs(np.linalg.norm(targets, axis=1) - 1.0),
            np.abs(np.linalg.norm(q_negative, axis=1) - 1.0),
        )
    )
    final_metadata = {
        **run_metadata,
        "completed": True,
        "embedding_dimension": int(q_positive.shape[1]),
        "sample_count": len(samples),
        "negative_count": len(negatives),
        "normalization_max_abs_error": float(norm_errors.max(initial=0.0)),
        "delta_identity_max_abs_error": max_identity_error,
        "reference_contribution_identity_max_abs_error": max_contribution_identity_error,
        "files": {
            "features": "features.npz",
            "rows": "rows.jsonl",
            "scores": "scores.csv",
        },
    }
    write_json_atomic(output_dir / "metadata.json", final_metadata)
    return final_metadata


def main() -> None:
    args = parse_args()
    started = time.time()
    model_path = (args.model_path or SUPPORTED_MODELS[args.model_name]).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint does not exist: {model_path}")
    batch_size = args.batch_size or DEFAULT_BATCH_SIZES[args.model_name]
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = args.output_dir / "cache"
    input_path = args.input.resolve()
    samples, negatives = load_dataset(input_path, args.limit_samples)
    run_metadata = {
        "schema_version": "1.1",
        "completed": False,
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "model_name": args.model_name,
        "model_path": str(model_path),
        "device": args.device,
        "dtype": args.dtype,
        "batch_size": batch_size,
        "limit_samples": args.limit_samples,
        "python": platform.python_version(),
    }
    if args.model_name == QWEN_MODEL_NAME:
        run_metadata["embedding_protocol"] = qwen_protocol_metadata()
    run_metadata["reference_baseline_protocol"] = reference_baseline_protocol(
        args.model_name
    )

    metadata_path = args.output_dir / "metadata.json"
    if metadata_path.is_file() and args.resume and not args.force:
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        compatibility_keys = ["input_sha256", "model_name", "model_path", "limit_samples"]
        if args.model_name == QWEN_MODEL_NAME:
            compatibility_keys.extend(["dtype", "embedding_protocol"])
        if "reference_baseline_protocol" in previous:
            compatibility_keys.append("reference_baseline_protocol")
        mismatches = [
            key for key in compatibility_keys if previous.get(key) != run_metadata.get(key)
        ]
        if mismatches:
            raise RuntimeError(
                "Existing output is incompatible for resume; use a different --output-dir "
                f"or --force. Mismatched fields: {mismatches}"
            )
    write_json_atomic(metadata_path, run_metadata)

    query_items = [
        QueryItem(
            cache_key=f"positive:{sample.sample_id}",
            query_id=f"p{sample.sample_index}",
            image_path=sample.reference_path,
            text=sample.positive_text,
        )
        for sample in samples
    ]
    query_items.extend(
        QueryItem(
            cache_key=f"negative:{item.negative_id}",
            query_id=f"n{item.negative_index}",
            image_path=samples[item.sample_index].reference_path,
            text=item.negative_text,
        )
        for item in negatives
    )

    print(
        f"[info] model={args.model_name} samples={len(samples)} "
        f"negatives={len(negatives)} batch_size={batch_size}"
    )
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    adapter = make_adapter(args, model_path)
    extract_targets(
        adapter,
        samples,
        cache_root,
        batch_size,
        resume=args.resume,
        force=args.force,
    )
    extract_reference_queries(
        adapter,
        args.model_name,
        samples,
        cache_root,
        batch_size,
        resume=args.resume,
        force=args.force,
    )
    extract_queries(
        adapter,
        query_items,
        cache_root,
        batch_size,
        resume=args.resume,
        force=args.force,
    )
    metadata = assemble_outputs(args.output_dir, cache_root, samples, negatives, run_metadata)
    metadata["elapsed_seconds"] = time.time() - started
    write_json_atomic(metadata_path, metadata)
    print(
        f"[done] {args.output_dir / 'features.npz'} "
        f"dimension={metadata['embedding_dimension']} elapsed={metadata['elapsed_seconds']:.1f}s"
    )


if __name__ == "__main__":
    main()
