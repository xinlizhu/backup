#!/usr/bin/env python3
"""Encode the S2 q- cache in disjoint shards across independent containers."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from cir_qdiag_train.on_policy_config import load_on_policy_config
from cir_qdiag_train.on_policy_data import (
    NegativeQueryLayout,
    load_cirr_positive_records,
    load_image_map,
    load_negative_query_layout,
)
from cir_qdiag_train.on_policy_mining import write_json_atomic
from prepare_on_policy_teacher_cache import (
    CIR_PLAY_ROOT,
    DEFAULT_TEACHER,
    _encode_negative_queries,
    _open_resumable_features,
    _save_array_atomic,
    _sha256,
    _write_negative_query_pair_scores,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("plan", "worker", "merge"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--teacher-model-name", default="BGE-VL-MLLM-S2")
    parser.add_argument("--teacher-model-path", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--query-batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.shard_count < 2:
        raise ValueError("shard-count must be at least two")
    if args.query_batch_size <= 0:
        raise ValueError("query-batch-size must be positive")
    context = _load_context(args.config, args.shard_count)
    if args.command == "plan":
        _create_plan(context)
    elif args.command == "worker":
        if args.shard_index is None:
            raise ValueError("worker requires --shard-index")
        _run_worker(args, context)
    else:
        _merge_shards(context)


def _load_context(config_path: Path, shard_count: int) -> dict[str, Any]:
    config = load_on_policy_config(config_path)
    if not config.mining.include_negative_queries_as_candidates:
        raise ValueError("config does not enable composed q- candidates")
    if config.data.teacher_cache_dir is None:
        raise ValueError("config has no Teacher cache directory")
    if config.data.negative_query_manifest is None:
        raise ValueError("config has no q- manifest")
    image_ids, image_paths = load_image_map(
        config.data.train_image_split_path,
        config.data.image_root,
    )
    records = load_cirr_positive_records(
        config.data.train_annotation_path,
        image_ids,
        image_paths,
    )
    layout = load_negative_query_layout(
        config.data.negative_query_manifest,
        records,
    )
    cache_dir = config.data.teacher_cache_dir
    return {
        "config": config,
        "records": records,
        "layout": layout,
        "cache_dir": cache_dir,
        "shard_dir": cache_dir / f"negative_query_shards_{shard_count:03d}",
        "manifest_sha256": _sha256(config.data.negative_query_manifest),
        "shard_count": shard_count,
    }


def _create_plan(context: dict[str, Any]) -> None:
    cache_dir: Path = context["cache_dir"]
    shard_dir: Path = context["shard_dir"]
    layout: NegativeQueryLayout = context["layout"]
    shard_count: int = context["shard_count"]
    progress_path = cache_dir / "negative_query_progress.json"
    features_path = cache_dir / "negative_query_features.npy"
    if not progress_path.is_file() or not features_path.is_file():
        raise RuntimeError(
            "no resumable q- cache was found; start the ordinary cache builder first"
        )
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    completed_prefix = int(progress.get("next_index", -1))
    total = len(layout.ids)
    if progress.get("count") != total or not 0 <= completed_prefix <= total:
        raise ValueError("global q- cache progress is incompatible with the manifest")
    # Catch an ordinary encoder that is still advancing the shared progress file.
    for _ in range(2):
        time.sleep(2)
        repeated = json.loads(progress_path.read_text(encoding="utf-8"))
        if int(repeated.get("next_index", -1)) != completed_prefix:
            raise RuntimeError(
                "the ordinary q- encoder is still running; stop it before planning shards"
            )
    features = np.load(features_path, mmap_mode="r")
    if features.ndim != 2 or features.shape[0] != total:
        raise ValueError("global q- feature cache has an invalid shape")
    remaining = total - completed_prefix
    base, extra = divmod(remaining, shard_count)
    ranges = []
    begin = completed_prefix
    for shard_index in range(shard_count):
        size = base + int(shard_index < extra)
        end = begin + size
        ranges.append({"shard_index": shard_index, "begin": begin, "end": end})
        begin = end
    plan = {
        "schema_version": 1,
        "negative_query_manifest_sha256": context["manifest_sha256"],
        "negative_query_count": total,
        "completed_prefix": completed_prefix,
        "feature_dimension": int(features.shape[1]),
        "feature_dtype": str(features.dtype),
        "shard_count": shard_count,
        "ranges": ranges,
    }
    plan_path = shard_dir / "plan.json"
    if plan_path.is_file():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing != plan:
            raise RuntimeError(
                f"an incompatible shard plan already exists: {plan_path}"
            )
        print(f"[reuse shard plan] {plan_path}", flush=True)
    else:
        shard_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(plan_path, plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)


def _run_worker(args: argparse.Namespace, context: dict[str, Any]) -> None:
    plan = _load_plan(context)
    shard_index = int(args.shard_index)
    if not 0 <= shard_index < plan["shard_count"]:
        raise ValueError(f"shard-index must be in [0, {plan['shard_count'] - 1}]")
    assigned = plan["ranges"][shard_index]
    begin = int(assigned["begin"])
    end = int(assigned["end"])
    count = end - begin
    if count <= 0:
        raise ValueError(f"shard {shard_index} has no assigned rows")
    if not args.teacher_model_path.is_dir():
        raise FileNotFoundError(args.teacher_model_path)
    if str(CIR_PLAY_ROOT) not in sys.path:
        sys.path.insert(0, str(CIR_PLAY_ROOT))
    from run_pinpoint_bge_lincir import choose_device, make_adapter, prepare_hf_runtime_cache

    shard_dir: Path = context["shard_dir"]
    worker_dir = shard_dir / f"shard_{shard_index:03d}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    prepare_hf_runtime_cache(worker_dir / "hf_runtime")
    device = choose_device(args.device)
    adapter = make_adapter(
        args.teacher_model_name,
        args.teacher_model_path.resolve(),
        device,
        args.dtype,
        hf_cache_dir=CIR_PLAY_ROOT / "models/checkpoints/lincir/.cache",
        lincir_clip_model_name="large",
    )
    np_dtype = np.dtype(plan["feature_dtype"])
    features = _open_resumable_features(
        path=worker_dir / "features.npy",
        progress_path=worker_dir / "progress.json",
        count=count,
        batch_size=args.query_batch_size,
        np_dtype=np_dtype,
        encode=lambda local_begin, local_end: _encode_negative_queries(
            adapter,
            context["records"],
            context["layout"],
            begin + local_begin,
            begin + local_end,
        ),
        description=(
            f"S2 q- shard {shard_index + 1}/{plan['shard_count']} "
            f"global[{begin}:{end}]"
        ),
    )
    if features.shape != (count, plan["feature_dimension"]):
        raise ValueError("completed shard feature shape is invalid")
    write_json_atomic(
        worker_dir / "complete.json",
        {
            "complete": True,
            "shard_index": shard_index,
            "begin": begin,
            "end": end,
            "count": count,
        },
    )
    print(f"[complete] shard {shard_index}: {worker_dir}", flush=True)
    adapter = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _merge_shards(context: dict[str, Any]) -> None:
    plan = _load_plan(context)
    cache_dir: Path = context["cache_dir"]
    shard_dir: Path = context["shard_dir"]
    layout: NegativeQueryLayout = context["layout"]
    global_path = cache_dir / "negative_query_features.npy"
    global_features = np.load(global_path, mmap_mode="r+")
    expected_global_shape = (
        plan["negative_query_count"],
        plan["feature_dimension"],
    )
    if global_features.shape != expected_global_shape:
        raise ValueError("global q- feature cache has an invalid shape")
    if str(global_features.dtype) != plan["feature_dtype"]:
        raise ValueError("global q- feature cache has an invalid dtype")
    copy_chunk = 8192
    for assigned in plan["ranges"]:
        shard_index = int(assigned["shard_index"])
        begin = int(assigned["begin"])
        end = int(assigned["end"])
        count = end - begin
        worker_dir = shard_dir / f"shard_{shard_index:03d}"
        complete_path = worker_dir / "complete.json"
        progress_path = worker_dir / "progress.json"
        features_path = worker_dir / "features.npy"
        if not all(
            path.is_file()
            for path in (complete_path, progress_path, features_path)
        ):
            raise RuntimeError(f"shard {shard_index} is incomplete: {worker_dir}")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("next_index") != count or progress.get("count") != count:
            raise RuntimeError(f"shard {shard_index} progress is incomplete")
        shard_features = np.load(features_path, mmap_mode="r")
        if shard_features.shape != (count, plan["feature_dimension"]):
            raise ValueError(f"shard {shard_index} feature shape is invalid")
        for local_begin in tqdm(
            range(0, count, copy_chunk),
            desc=f"merge q- shard {shard_index + 1}/{plan['shard_count']}",
        ):
            local_end = min(local_begin + copy_chunk, count)
            global_features[begin + local_begin : begin + local_end] = (
                shard_features[local_begin:local_end]
            )
        global_features.flush()

    write_json_atomic(
        cache_dir / "negative_query_progress.json",
        {"next_index": len(layout.ids), "count": len(layout.ids)},
    )
    _save_array_atomic(cache_dir / "negative_query_owners.npy", layout.owners)
    write_json_atomic(cache_dir / "negative_query_ids.json", layout.ids)
    positive_queries = np.load(cache_dir / "query_features.npy", mmap_mode="r")
    _write_negative_query_pair_scores(
        cache_dir / "negative_query_pair_scores.npy",
        positive_queries=positive_queries,
        negative_queries=global_features,
        owners=layout.owners,
    )
    metadata_path = cache_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "complete": True,
            "negative_query_candidates": True,
            "negative_query_manifest_sha256": context["manifest_sha256"],
            "negative_query_count": len(layout.ids),
            "negative_query_embedding_dimension": int(global_features.shape[1]),
            "negative_query_shard_count": plan["shard_count"],
            "negative_query_shard_completed_prefix": plan["completed_prefix"],
        }
    )
    write_json_atomic(metadata_path, metadata)
    write_json_atomic(
        cache_dir / "negative_query_layout.json",
        {
            "schema_version": 1,
            "negative_query_manifest_sha256": context["manifest_sha256"],
            "negative_query_count": len(layout.ids),
        },
    )
    print(
        f"[merged] complete q- Teacher cache: {cache_dir} "
        f"({len(layout.ids)} rows)",
        flush=True,
    )


def _load_plan(context: dict[str, Any]) -> dict[str, Any]:
    plan_path: Path = context["shard_dir"] / "plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError(f"create the shard plan first: {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    expected = {
        "negative_query_manifest_sha256": context["manifest_sha256"],
        "negative_query_count": len(context["layout"].ids),
        "shard_count": context["shard_count"],
    }
    if any(plan.get(key) != value for key, value in expected.items()):
        raise ValueError("q- shard plan does not match the configured data")
    return plan


if __name__ == "__main__":
    main()
