#!/usr/bin/env python3
"""Evaluate cached composed-query and gallery features on full CIRR train."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from cir_qdiag_train.on_policy_config import load_on_policy_config
from cir_qdiag_train.on_policy_data import load_cirr_positive_records, load_image_map
from cir_qdiag_train.on_policy_mining import write_json_atomic


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs/on_policy_clip_l_rank.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--gallery-features", type=Path, default=None)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-batch-size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.score_batch_size <= 0:
        raise ValueError("--score-batch-size must be positive")
    config = load_on_policy_config(args.config)
    image_ids, image_paths = load_image_map(
        config.data.train_image_split_path, config.data.image_root
    )
    records = load_cirr_positive_records(
        config.data.train_annotation_path, image_ids, image_paths
    )
    cache_dir = args.cache_dir.resolve()
    gallery_path = (
        args.gallery_features.resolve()
        if args.gallery_features is not None
        else cache_dir / "gallery_features.npy"
    )
    query_path = cache_dir / "query_features.npy"
    gallery = np.load(gallery_path, mmap_mode="r")
    queries = np.load(query_path, mmap_mode="r")
    if gallery.ndim != 2 or queries.ndim != 2:
        raise ValueError("Feature caches must be rank-2 arrays")
    if gallery.shape[0] != len(image_ids) or queries.shape[0] != len(records):
        raise ValueError(
            f"Cache counts do not match CIRR train: gallery={gallery.shape}, "
            f"queries={queries.shape}"
        )
    if gallery.shape[1] != queries.shape[1]:
        raise ValueError("Gallery/query embedding dimensions do not match")
    for name, expected in (
        ("image_ids.json", image_ids),
        ("pair_ids.json", [record.pair_id for record in records]),
    ):
        path = cache_dir / name
        if path.is_file() and json.loads(path.read_text(encoding="utf-8")) != expected:
            raise ValueError(f"Cache ordering mismatch: {path}")

    image_index = {image_id: index for index, image_id in enumerate(image_ids)}
    references = np.asarray(
        [image_index[record.reference_id] for record in records], dtype=np.int64
    )
    targets = np.asarray(
        [image_index[record.target_id] for record in records], dtype=np.int64
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
    gallery_device = torch.from_numpy(
        np.asarray(gallery, dtype=np.float32)
    ).to(device)
    ks = (1, 5, 10, 32, 50)
    hits = {k: 0 for k in ks}
    ranks = np.empty(len(records), dtype=np.int32)
    for begin in tqdm(
        range(0, len(records), args.score_batch_size),
        desc=f"{args.model_label} full-gallery CIRR-train recall",
    ):
        end = min(begin + args.score_batch_size, len(records))
        query = torch.from_numpy(
            np.asarray(queries[begin:end], dtype=np.float32)
        ).to(device)
        scores = query @ gallery_device.t()
        local_reference = torch.from_numpy(references[begin:end]).to(device)
        local_target = torch.from_numpy(targets[begin:end]).to(device)
        scores.scatter_(1, local_reference[:, None], -torch.inf)
        target_scores = scores.gather(1, local_target[:, None])
        local_ranks = 1 + (scores > target_scores).sum(dim=1)
        ranks[begin:end] = local_ranks.cpu().numpy().astype(np.int32)
        for k in ks:
            hits[k] += int((local_ranks <= k).sum().item())

    result = {
        "model": args.model_label,
        "protocol": "CIRR train full-gallery target retrieval; reference excluded",
        "queries": len(records),
        "gallery_images": len(image_ids),
        **{f"recall_at_{k}": 100.0 * hits[k] / len(records) for k in ks},
        "mean_rank": float(ranks.mean()),
        "median_rank": float(np.median(ranks)),
        "gallery_features": str(gallery_path),
        "query_features": str(query_path),
    }
    np.save(cache_dir / "train_target_ranks.npy", ranks)
    write_json_atomic(cache_dir / "train_recall.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
