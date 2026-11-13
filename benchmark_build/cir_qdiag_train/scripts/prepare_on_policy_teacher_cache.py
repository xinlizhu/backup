#!/usr/bin/env python3
"""Cache frozen Teacher image/query features for CIRR on-policy KD."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from cir_qdiag_train.on_policy_config import load_on_policy_config
from cir_qdiag_train.on_policy_data import (
    CIRRPositiveRecord,
    NegativeQueryLayout,
    load_cirr_positive_records,
    load_image_map,
    load_negative_query_layout,
)
from cir_qdiag_train.on_policy_mining import write_json_atomic


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
CIR_PLAY_ROOT = WORKSPACE_ROOT / "CIR_PLAY"
DEFAULT_TEACHER = CIR_PLAY_ROOT / "models/checkpoints/BGE-VL-MLLM-S2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--teacher-model-name", default="BGE-VL-MLLM-S2")
    parser.add_argument("--teacher-model-path", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override data.teacher_cache_dir so the same cache builder can be used for baselines.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--image-batch-size", type=int, default=16)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.image_batch_size <= 0 or args.query_batch_size <= 0:
        raise ValueError("Teacher cache batch sizes must be positive")
    config = load_on_policy_config(args.config)
    image_ids, image_paths = load_image_map(
        config.data.train_image_split_path, config.data.image_root
    )
    records = load_cirr_positive_records(
        config.data.train_annotation_path, image_ids, image_paths
    )
    negative_layout = None
    negative_manifest_sha256 = None
    if config.mining.include_negative_queries_as_candidates:
        if config.data.negative_query_manifest is None:
            raise ValueError("mixed q- candidates require a negative-query manifest")
        negative_layout = load_negative_query_layout(
            config.data.negative_query_manifest,
            records,
        )
        negative_manifest_sha256 = _sha256(config.data.negative_query_manifest)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else config.data.teacher_cache_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    expected = {
        "schema_version": 1,
        "teacher_model_name": args.teacher_model_name,
        "teacher_model_path": str(args.teacher_model_path.resolve()),
        "gallery_count": len(image_ids),
        "query_count": len(records),
        "feature_dtype": config.mining.cache_dtype,
    }
    negative_layout_state = None
    if negative_layout is not None:
        negative_layout_state = {
            "schema_version": 1,
            "negative_query_manifest_sha256": negative_manifest_sha256,
            "negative_query_count": len(negative_layout.ids),
        }
        expected.update({"negative_query_candidates": True, **negative_layout_state})
    if metadata_path.is_file() and not args.force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        required = [
            output_dir / "gallery_features.npy",
            output_dir / "query_features.npy",
            output_dir / "image_ids.json",
            output_dir / "pair_ids.json",
        ]
        if negative_layout is not None:
            required.extend(
                [
                    output_dir / "negative_query_features.npy",
                    output_dir / "negative_query_layout.json",
                    output_dir / "negative_query_ids.json",
                    output_dir / "negative_query_owners.npy",
                    output_dir / "negative_query_pair_scores.npy",
                ]
            )
        if (
            metadata.get("complete")
            and all(metadata.get(key) == value for key, value in expected.items())
            and all(path.is_file() for path in required)
        ):
            print(f"[skip] complete Teacher cache: {output_dir}", flush=True)
            return
        if negative_layout_state is not None:
            layout_path = output_dir / "negative_query_layout.json"
            cached_layout = (
                json.loads(layout_path.read_text(encoding="utf-8"))
                if layout_path.is_file()
                else None
            )
            if cached_layout != negative_layout_state:
                _remove_negative_query_cache(output_dir)
    if args.force:
        for name in (
            "gallery_features.npy",
            "query_features.npy",
            "gallery_progress.json",
            "query_progress.json",
            "image_ids.json",
            "pair_ids.json",
            "metadata.json",
        ):
            (output_dir / name).unlink(missing_ok=True)
        _remove_negative_query_cache(output_dir)
    if not args.teacher_model_path.is_dir():
        raise FileNotFoundError(args.teacher_model_path)
    if negative_layout_state is not None:
        write_json_atomic(
            output_dir / "negative_query_layout.json",
            negative_layout_state,
        )

    if str(CIR_PLAY_ROOT) not in sys.path:
        sys.path.insert(0, str(CIR_PLAY_ROOT))
    from run_pinpoint_bge_lincir import choose_device, make_adapter, prepare_hf_runtime_cache

    prepare_hf_runtime_cache(output_dir / "hf_runtime")
    device = choose_device(args.device)
    adapter = make_adapter(
        args.teacher_model_name,
        args.teacher_model_path.resolve(),
        device,
        args.dtype,
        hf_cache_dir=CIR_PLAY_ROOT / "models/checkpoints/lincir/.cache",
        lincir_clip_model_name="large",
    )
    started = time.perf_counter()
    np_dtype = np.float16 if config.mining.cache_dtype == "float16" else np.float32
    gallery = _open_resumable_features(
        path=output_dir / "gallery_features.npy",
        progress_path=output_dir / "gallery_progress.json",
        count=len(image_ids),
        batch_size=args.image_batch_size,
        np_dtype=np_dtype,
        encode=lambda begin, end: adapter.encode_image_batch(image_paths[begin:end]),
        description=f"{args.teacher_model_name} CIRR train gallery",
    )
    queries = _open_resumable_features(
        path=output_dir / "query_features.npy",
        progress_path=output_dir / "query_progress.json",
        count=len(records),
        batch_size=args.query_batch_size,
        np_dtype=np_dtype,
        encode=lambda begin, end: _encode_queries(adapter, records[begin:end], begin),
        description=f"{args.teacher_model_name} CIRR train queries",
    )
    negative_query_dimension = None
    if negative_layout is not None:
        negative_queries = _open_resumable_features(
            path=output_dir / "negative_query_features.npy",
            progress_path=output_dir / "negative_query_progress.json",
            count=len(negative_layout.ids),
            batch_size=args.query_batch_size,
            np_dtype=np_dtype,
            encode=lambda begin, end: _encode_negative_queries(
                adapter,
                records,
                negative_layout,
                begin,
                end,
            ),
            description=f"{args.teacher_model_name} CIR-QDiag q- candidates",
        )
        negative_query_dimension = int(negative_queries.shape[1])
        _save_array_atomic(
            output_dir / "negative_query_owners.npy",
            negative_layout.owners,
        )
        write_json_atomic(
            output_dir / "negative_query_ids.json",
            negative_layout.ids,
        )
        _write_negative_query_pair_scores(
            output_dir / "negative_query_pair_scores.npy",
            positive_queries=queries,
            negative_queries=negative_queries,
            owners=negative_layout.owners,
        )
    write_json_atomic(output_dir / "image_ids.json", image_ids)
    write_json_atomic(output_dir / "pair_ids.json", [record.pair_id for record in records])
    metadata = {
        **expected,
        "complete": True,
        "embedding_dimension": int(gallery.shape[1]),
        "query_embedding_dimension": int(queries.shape[1]),
        "negative_query_embedding_dimension": negative_query_dimension,
        "elapsed_seconds_this_process": time.perf_counter() - started,
    }
    write_json_atomic(metadata_path, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    adapter = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _encode_queries(adapter, records: list[CIRRPositiveRecord], begin: int) -> np.ndarray:
    frame = pd.DataFrame(
        {
            "query_id": [f"on_policy_{begin + offset}" for offset in range(len(records))],
            "query_image_signature": [str(record.reference_path) for record in records],
            "instruction": [record.text for record in records],
        }
    )
    return np.asarray(adapter.encode_query_batch(frame, image_cache=Path("/")))


def _encode_negative_queries(
    adapter,
    records: list[CIRRPositiveRecord],
    layout: NegativeQueryLayout,
    begin: int,
    end: int,
) -> np.ndarray:
    owners = layout.owners[begin:end].tolist()
    frame = pd.DataFrame(
        {
            "query_id": layout.ids[begin:end],
            "query_image_signature": [
                str(records[int(owner)].reference_path) for owner in owners
            ],
            "instruction": layout.texts[begin:end],
        }
    )
    return np.asarray(adapter.encode_query_batch(frame, image_cache=Path("/")))


def _write_negative_query_pair_scores(
    path: Path,
    *,
    positive_queries: np.ndarray,
    negative_queries: np.ndarray,
    owners: np.ndarray,
) -> None:
    expected_shape = (len(owners),)
    if path.is_file():
        existing = np.load(path, mmap_mode="r")
        if existing.shape == expected_shape and existing.dtype == np.float32:
            return
        path.unlink()
    scores = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=expected_shape,
    )
    chunk_size = 4096
    for begin in tqdm(
        range(0, len(owners), chunk_size),
        desc="S2 q+ to q- pair scores",
    ):
        end = min(begin + chunk_size, len(owners))
        positive = np.asarray(
            positive_queries[owners[begin:end]], dtype=np.float32
        )
        negative = np.asarray(negative_queries[begin:end], dtype=np.float32)
        scores[begin:end] = np.einsum("bd,bd->b", positive, negative)
    scores.flush()


def _remove_negative_query_cache(output_dir: Path) -> None:
    for name in (
        "negative_query_features.npy",
        "negative_query_progress.json",
        "negative_query_layout.json",
        "negative_query_ids.json",
        "negative_query_owners.npy",
        "negative_query_pair_scores.npy",
    ):
        (output_dir / name).unlink(missing_ok=True)


def _save_array_atomic(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_resumable_features(
    *,
    path: Path,
    progress_path: Path,
    count: int,
    batch_size: int,
    np_dtype,
    encode: Callable[[int, int], np.ndarray],
    description: str,
) -> np.memmap:
    start = 0
    features: np.memmap | None = None
    if path.is_file() or progress_path.is_file():
        if not path.is_file() or not progress_path.is_file():
            raise RuntimeError(f"incomplete cache for {description}; rerun with --force")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        features = np.load(path, mmap_mode="r+")
        start = int(progress.get("next_index", 0))
        if features.ndim != 2 or features.shape[0] != count or start > count:
            raise ValueError(f"invalid cache state for {description}")
    if features is None:
        first_end = min(batch_size, count)
        first = _normalize(np.asarray(encode(0, first_end), dtype=np.float32))
        if first.ndim != 2 or first.shape[0] != first_end:
            raise ValueError(f"Teacher returned invalid features for {description}: {first.shape}")
        features = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=np_dtype,
            shape=(count, first.shape[1]),
        )
        features[:first_end] = first.astype(np_dtype)
        features.flush()
        start = first_end
        write_json_atomic(progress_path, {"next_index": start, "count": count})
    for begin in tqdm(
        range(start, count, batch_size),
        total=(count + batch_size - 1) // batch_size,
        initial=start // batch_size,
        desc=description,
    ):
        end = min(begin + batch_size, count)
        encoded = _normalize(np.asarray(encode(begin, end), dtype=np.float32))
        if encoded.shape != (end - begin, features.shape[1]):
            raise ValueError(f"Teacher returned invalid features for {description}: {encoded.shape}")
        features[begin:end] = encoded.astype(np_dtype)
        features.flush()
        write_json_atomic(progress_path, {"next_index": end, "count": count})
    return features


def _normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or bool((norms <= 1e-12).any()):
        raise ValueError("Teacher produced non-finite or zero-norm features")
    return values / norms


if __name__ == "__main__":
    main()
