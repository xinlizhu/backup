#!/usr/bin/env python3
"""Evaluate a CIR_PLAY original-model adapter on FashionIQ val."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cir_qdiag_train.fashioniq_evaluation import (
    FASHIONIQ_CATEGORIES,
    aggregate_category_metrics,
    cache_feature_matrix,
    evaluate_feature_files,
    load_fashioniq_val,
)
from cir_qdiag_train.on_policy_mining import write_json_atomic


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
CIR_PLAY_ROOT = WORKSPACE_ROOT / "CIR_PLAY"
DEFAULT_DATASET = WORKSPACE_ROOT / "CIR_new/datasets/fashion-iq"
DEFAULT_MODEL = CIR_PLAY_ROOT / "models/checkpoints/BGE-VL-MLLM-S2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="BGE-VL-MLLM-S2 Teacher")
    parser.add_argument("--adapter-name", default="BGE-VL-MLLM-S2")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--image-batch-size", type=int, default=16)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--score-batch-size", type=int, default=1024)
    parser.add_argument("--force-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in ("image_batch_size", "query_batch_size", "score_batch_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    model_path = args.model_path.resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema_version": 1,
        "model_name": args.model_name,
        "adapter_name": args.adapter_name,
        "model_path": str(model_path),
        "dataset_root": str(args.dataset_root.resolve()),
        "dtype": args.dtype,
        "device": args.device,
    }
    _prepare_output(output_dir, identity, force=args.force_cache)
    final_path = output_dir / "metrics.json"
    if final_path.is_file() and not args.force_cache:
        print(final_path.read_text(encoding="utf-8"), flush=True)
        return

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
        args.adapter_name,
        model_path,
        device,
        args.dtype,
        hf_cache_dir=CIR_PLAY_ROOT / "models/checkpoints/lincir/.cache/huggingface",
        lincir_clip_model_name="large",
    )
    category_metrics: dict[str, dict[str, float | int]] = {}
    for category in FASHIONIQ_CATEGORIES:
        gallery_ids, gallery_paths, records = load_fashioniq_val(
            args.dataset_root, category
        )
        category_dir = output_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        gallery_path = category_dir / "gallery_features.npy"
        query_path = category_dir / "query_features.npy"
        cache_feature_matrix(
            gallery_path,
            category_dir / "gallery_progress.json",
            count=len(gallery_paths),
            batch_size=args.image_batch_size,
            encode=lambda begin, end, paths=gallery_paths: adapter.encode_image_batch(
                paths[begin:end]
            ),
            description=f"{args.model_name} FashionIQ-{category} gallery",
        )
        cache_feature_matrix(
            query_path,
            category_dir / "query_progress.json",
            count=len(records),
            batch_size=args.query_batch_size,
            encode=lambda begin, end, rows=records: _encode_queries(
                adapter,
                rows[begin:end],
                args.dataset_root,
                category,
                begin,
            ),
            description=f"{args.model_name} FashionIQ-{category} queries",
        )
        metrics = evaluate_feature_files(
            gallery_path=gallery_path,
            query_path=query_path,
            gallery_ids=gallery_ids,
            records=records,
            device=device,
            score_batch_size=args.score_batch_size,
        )
        write_json_atomic(category_dir / "metrics.json", metrics)
        category_metrics[category] = metrics

    result = aggregate_category_metrics(
        model_name=args.model_name,
        model_source=str(model_path),
        category_metrics=category_metrics,
    )
    write_json_atomic(final_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    adapter = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _encode_queries(
    adapter,
    records,
    dataset_root: Path,
    category: str,
    begin: int,
) -> np.ndarray:
    image_root = dataset_root.resolve() / "images"
    frame = pd.DataFrame(
        {
            "query_id": [
                f"fashioniq_{category}_{begin + index}" for index in range(len(records))
            ],
            "query_image_signature": [
                str(image_root / f"{record.reference_id}.png") for record in records
            ],
            "instruction": [record.text for record in records],
        }
    )
    return np.asarray(adapter.encode_query_batch(frame, image_cache=Path("/")))


def _prepare_output(output_dir: Path, identity: dict, *, force: bool) -> None:
    metadata_path = output_dir / "model_metadata.json"
    if force:
        for category in FASHIONIQ_CATEGORIES:
            for name in (
                "gallery_features.npy",
                "gallery_progress.json",
                "query_features.npy",
                "query_progress.json",
                "metrics.json",
            ):
                (output_dir / category / name).unlink(missing_ok=True)
        (output_dir / "metrics.json").unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
    if metadata_path.is_file():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous != identity:
            raise RuntimeError(
                f"output cache belongs to a different model: {output_dir}; "
                "choose another --output-dir or pass --force-cache"
            )
    else:
        write_json_atomic(metadata_path, identity)


if __name__ == "__main__":
    main()
