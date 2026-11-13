#!/usr/bin/env python3
"""Evaluate an original CIR_PLAY-compatible model on CIRR validation."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Callable, get_args

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from cir_qdiag_train.on_policy_config import load_on_policy_config
from cir_qdiag_train.on_policy_data import load_cirr_positive_records, load_image_map
from cir_qdiag_train.on_policy_mining import write_json_atomic
from cir_qdiag_train.qwen_on_policy_config import QwenQueryLoRAConfig, load_qwen_on_policy_config


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
CIR_PLAY_ROOT = WORKSPACE_ROOT / "CIR_PLAY"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/on_policy_clip_l_rank.yaml"
DEFAULT_MODEL = CIR_PLAY_ROOT / "models/checkpoints/BGE-VL-MLLM-S2"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/on_policy_live_top10/teacher/BGE-VL-MLLM-S2/cirr_val"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-name", default="BGE-VL-MLLM-S2")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--image-batch-size", type=int, default=16)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--score-batch-size", type=int, default=512)
    parser.add_argument("--force-cache", action="store_true")
    return parser.parse_args()


def load_evaluation_config(path: Path):
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    # Derive supported types from the schema, rather than maintaining a second
    # model-type list that can silently route a new Qwen variant into CLIP.
    variants = get_args(QwenQueryLoRAConfig.model_fields["variant"].annotation)
    if payload.get("model", {}).get("variant") in variants:
        return load_qwen_on_policy_config(path)
    return load_on_policy_config(path)


def main() -> None:
    args = parse_args()
    for name in ("image_batch_size", "query_batch_size", "score_batch_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    config = load_evaluation_config(args.config)
    image_ids, image_paths = load_image_map(
        config.data.val_image_split_path, config.data.image_root
    )
    records = load_cirr_positive_records(
        config.data.val_annotation_path, image_ids, image_paths
    )
    annotations = json.loads(config.data.val_annotation_path.read_text(encoding="utf-8"))
    if len(annotations) != len(records):
        raise ValueError("CIRR val annotation/record counts do not match")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_path.resolve()
    expected = {
        "schema_version": 1,
        "complete": True,
        "model_name": args.model_name,
        "model_path": str(model_path),
        "gallery_count": len(image_ids),
        "query_count": len(records),
        "feature_dtype": "float32",
    }
    metadata_path = output_dir / "cache_metadata.json"
    gallery_path = output_dir / "gallery_features.npy"
    query_path = output_dir / "query_features.npy"
    cache_complete = False
    if metadata_path.is_file() and not args.force_cache:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        cache_complete = (
            all(metadata.get(key) == value for key, value in expected.items())
            and gallery_path.is_file()
            and query_path.is_file()
        )

    if not cache_complete:
        if args.force_cache:
            for name in (
                "gallery_features.npy",
                "gallery_progress.json",
                "query_features.npy",
                "query_progress.json",
                "cache_metadata.json",
            ):
                (output_dir / name).unlink(missing_ok=True)
        if str(CIR_PLAY_ROOT) not in sys.path:
            sys.path.insert(0, str(CIR_PLAY_ROOT))
        from run_pinpoint_bge_lincir import (
            choose_device,
            make_adapter,
            prepare_hf_runtime_cache,
        )

        prepare_hf_runtime_cache(output_dir / "hf_runtime")
        device = choose_device(args.device)
        adapter = make_adapter(
            args.model_name,
            model_path,
            device,
            args.dtype,
            hf_cache_dir=CIR_PLAY_ROOT / "models/checkpoints/lincir/.cache/huggingface",
            lincir_clip_model_name="large",
        )
        _cache_features(
            gallery_path,
            output_dir / "gallery_progress.json",
            count=len(image_paths),
            batch_size=args.image_batch_size,
            encode=lambda begin, end: adapter.encode_image_batch(
                image_paths[begin:end]
            ),
            description=f"{args.model_name} CIRR-val gallery",
        )
        _cache_features(
            query_path,
            output_dir / "query_progress.json",
            count=len(records),
            batch_size=args.query_batch_size,
            encode=lambda begin, end: _encode_queries(
                adapter, records[begin:end], begin
            ),
            description=f"{args.model_name} CIRR-val queries",
        )
        write_json_atomic(output_dir / "image_ids.json", image_ids)
        write_json_atomic(
            output_dir / "pair_ids.json", [record.pair_id for record in records]
        )
        gallery = np.load(gallery_path, mmap_mode="r")
        write_json_atomic(
            metadata_path,
            {**expected, "embedding_dimension": int(gallery.shape[1])},
        )
        adapter = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics = _evaluate(
        model_name=args.model_name,
        model_path=model_path,
        records=records,
        annotations=annotations,
        image_ids=image_ids,
        gallery_path=gallery_path,
        query_path=query_path,
        device_name=args.device,
        score_batch_size=args.score_batch_size,
    )
    write_json_atomic(output_dir / "val_metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


def _encode_queries(adapter, records, begin: int) -> np.ndarray:
    frame = pd.DataFrame(
        {
            "query_id": [f"cirr_val_{begin + i}" for i in range(len(records))],
            "query_image_signature": [
                str(record.reference_path) for record in records
            ],
            "instruction": [record.text for record in records],
        }
    )
    return np.asarray(adapter.encode_query_batch(frame, image_cache=Path("/")))


def _cache_features(
    path: Path,
    progress_path: Path,
    *,
    count: int,
    batch_size: int,
    encode: Callable[[int, int], np.ndarray],
    description: str,
) -> None:
    start = 0
    features: np.memmap | None = None
    if path.is_file() or progress_path.is_file():
        if not path.is_file() or not progress_path.is_file():
            raise RuntimeError(f"incomplete cache for {description}; use --force-cache")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        features = np.load(path, mmap_mode="r+")
        start = int(progress.get("next_index", 0))
        if features.ndim != 2 or features.shape[0] != count or start > count:
            raise ValueError(f"invalid resumable cache for {description}")
    for begin in tqdm(
        range(start, count, batch_size),
        total=(count + batch_size - 1) // batch_size,
        initial=start // batch_size,
        desc=description,
    ):
        end = min(begin + batch_size, count)
        encoded = _normalize(np.asarray(encode(begin, end), dtype=np.float32))
        if encoded.ndim != 2 or encoded.shape[0] != end - begin:
            raise ValueError(f"invalid features for {description}: {encoded.shape}")
        if features is None:
            features = np.lib.format.open_memmap(
                path,
                mode="w+",
                dtype=np.float32,
                shape=(count, encoded.shape[1]),
            )
        if encoded.shape[1] != features.shape[1]:
            raise ValueError(f"embedding dimension changed for {description}")
        features[begin:end] = encoded
        features.flush()
        write_json_atomic(
            progress_path, {"next_index": end, "count": count}
        )
    if features is None:
        raise ValueError(f"cannot build an empty cache for {description}")


def _normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or bool((norms <= 1e-12).any()):
        raise ValueError("model produced non-finite or zero-norm features")
    return values / norms


def _evaluate(
    *,
    model_name: str,
    model_path: Path,
    records,
    annotations: list[dict],
    image_ids: list[str],
    gallery_path: Path,
    query_path: Path,
    device_name: str,
    score_batch_size: int,
) -> dict[str, float | int | str]:
    gallery = np.load(gallery_path, mmap_mode="r")
    queries = np.load(query_path, mmap_mode="r")
    if gallery.shape[0] != len(image_ids) or queries.shape[0] != len(records):
        raise ValueError("CIRR val cache counts do not match annotations")
    image_index = {image_id: index for index, image_id in enumerate(image_ids)}
    references = np.asarray(
        [image_index[record.reference_id] for record in records], dtype=np.int64
    )
    targets = np.asarray(
        [image_index[record.target_id] for record in records], dtype=np.int64
    )
    device = torch.device(
        device_name if torch.cuda.is_available() else "cpu"
    )
    gallery_device = torch.from_numpy(
        np.asarray(gallery, dtype=np.float32)
    ).to(device)
    recall_k = (1, 5, 10, 50)
    group_k = (1, 2, 3)
    recall_hits = {k: 0 for k in recall_k}
    group_hits = {k: 0 for k in group_k}
    for begin in tqdm(
        range(0, len(records), score_batch_size),
        desc=f"{model_name} CIRR-val scoring",
    ):
        end = min(begin + score_batch_size, len(records))
        query = torch.from_numpy(
            np.asarray(queries[begin:end], dtype=np.float32)
        ).to(device)
        scores = query @ gallery_device.t()
        local_references = torch.from_numpy(references[begin:end]).to(device)
        scores.scatter_(1, local_references[:, None], -torch.inf)
        top = torch.topk(scores, k=max(recall_k), dim=1).indices.cpu().numpy()
        for offset, row in enumerate(annotations[begin:end]):
            target_index = int(targets[begin + offset])
            for k in recall_k:
                recall_hits[k] += target_index in top[offset, :k]
            members = row.get("img_set", {}).get("members", [])
            group_ids = [
                image_id
                for image_id in members
                if image_id != records[begin + offset].reference_id
                and image_id in image_index
            ]
            if records[begin + offset].target_id not in group_ids:
                raise ValueError("CIRR target is absent from its validation group")
            group_indices = torch.tensor(
                [image_index[image_id] for image_id in group_ids],
                device=device,
            )
            group_scores = scores[offset, group_indices]
            order = group_scores.argsort(descending=True).cpu().tolist()
            ranked_ids = [group_ids[index] for index in order]
            for k in group_k:
                group_hits[k] += records[begin + offset].target_id in ranked_ids[:k]
    count = len(records)
    return {
        "model": model_name,
        "checkpoint": str(model_path),
        "num_queries": count,
        **{f"recall_at_{k}": round(100.0 * recall_hits[k] / count, 4) for k in recall_k},
        **{
            f"group_recall_at_{k}": round(100.0 * group_hits[k] / count, 4)
            for k in group_k
        },
    }


if __name__ == "__main__":
    main()
