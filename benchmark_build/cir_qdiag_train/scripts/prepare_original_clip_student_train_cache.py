#!/usr/bin/env python3
"""Reconstruct and cache the epoch-1 (global-step-0) CLIP Student queries.

The epoch-1 on-policy directory already contains the initial Student gallery.
This script deterministically reconstructs the randomly initialized CIR fusion
head with the experiment seed, then composes all CIRR-train queries against the
cached reference-image features.  It is resumable at query-batch boundaries.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from cir_qdiag_train.model import ComposedRetrievalModel
from cir_qdiag_train.on_policy_config import load_on_policy_config
from cir_qdiag_train.on_policy_data import load_cirr_positive_records, load_image_map
from cir_qdiag_train.on_policy_mining import write_json_atomic


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs/on_policy_clip_l_rank.yaml"
DEFAULT_GALLERY = (
    PROJECT_ROOT
    / "outputs/on_policy_centered/clip_l_original/positive_rank/on_policy/epoch_001/gallery_features.npy"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/cirr_train_recall/clip_l_initial_student"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gallery-features", type=Path, default=DEFAULT_GALLERY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--query-batch-size", type=int, default=1024)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.query_batch_size <= 0:
        raise ValueError("--query-batch-size must be positive")
    config = load_on_policy_config(args.config)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    query_path = output_dir / "query_features.npy"
    progress_path = output_dir / "query_progress.json"
    metadata_path = output_dir / "metadata.json"
    if args.force:
        for path in (query_path, progress_path, metadata_path):
            path.unlink(missing_ok=True)

    image_ids, image_paths = load_image_map(
        config.data.train_image_split_path, config.data.image_root
    )
    records = load_cirr_positive_records(
        config.data.train_annotation_path, image_ids, image_paths
    )
    gallery_path = args.gallery_features.resolve()
    gallery = np.load(gallery_path, mmap_mode="r")
    if gallery.ndim != 2 or gallery.shape[0] != len(image_ids):
        raise ValueError(
            f"Initial Student gallery shape {gallery.shape} does not match "
            f"{len(image_ids)} CIRR train images"
        )
    image_index = {image_id: index for index, image_id in enumerate(image_ids)}
    reference_indices = np.asarray(
        [image_index[record.reference_id] for record in records], dtype=np.int64
    )

    expected = {
        "schema_version": 1,
        "complete": True,
        "model_label": "CLIP-L initial Student (epoch 1, global step 0)",
        "config": str(args.config.resolve()),
        "seed": config.experiment.seed,
        "gallery_features": str(gallery_path),
        "gallery_count": len(image_ids),
        "query_count": len(records),
        "embedding_dimension": int(gallery.shape[1]),
        "feature_dtype": "float32",
    }
    if metadata_path.is_file() and not args.force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if all(metadata.get(key) == value for key, value in expected.items()):
            print(f"[skip] complete initial Student query cache: {output_dir}", flush=True)
            return

    _set_seed(config.experiment.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
    model, processor = ComposedRetrievalModel.from_pretrained(
        config.model.pretrained_model_name_or_path,
        variant=config.model.variant,
        local_files_only=config.model.local_files_only,
        trust_remote_code=config.model.trust_remote_code,
        composition_head=config.model.composition_head,
        fusion_init_path=config.model.fusion_init_path,
        hidden_dim=config.model.hidden_dim,
        dropout=config.model.dropout,
        freeze_backbone=config.model.freeze_backbone,
        freeze_vision_encoder=config.model.freeze_vision_encoder,
        gradient_checkpointing=config.model.gradient_checkpointing,
        temperature=config.optimizer.temperature,
        learnable_temperature=config.optimizer.learnable_temperature,
        unfreeze_last_n_layers=config.model.unfreeze_last_n_layers,
        lora_rank=config.model.lora_rank,
        lora_alpha=config.model.lora_alpha,
        lora_dropout=config.model.lora_dropout,
    )
    model.to(device).eval()
    gallery_device = torch.from_numpy(
        np.asarray(gallery, dtype=np.float32)
    ).to(device)

    start = 0
    queries: np.memmap | None = None
    if query_path.is_file() or progress_path.is_file():
        if not query_path.is_file() or not progress_path.is_file():
            raise RuntimeError("Incomplete initial Student cache; rerun with --force")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        queries = np.load(query_path, mmap_mode="r+")
        start = int(progress.get("next_index", 0))
        if queries.shape != (len(records), gallery.shape[1]) or start > len(records):
            raise ValueError("Invalid initial Student query-cache state")
    if queries is None:
        queries = np.lib.format.open_memmap(
            query_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(records), gallery.shape[1]),
        )

    for begin in tqdm(
        range(start, len(records), args.query_batch_size),
        total=(len(records) + args.query_batch_size - 1) // args.query_batch_size,
        initial=start // args.query_batch_size,
        desc="CLIP-L initial Student CIRR-train queries",
    ):
        end = min(begin + args.query_batch_size, len(records))
        inputs = processor(
            text=[record.text for record in records[begin:end]],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with torch.inference_mode(), _autocast(device, config.optimizer.mixed_precision):
            text_features = model.encode_text(
                inputs["input_ids"].to(device),
                (
                    None
                    if inputs.get("attention_mask") is None
                    else inputs["attention_mask"].to(device)
                ),
            )
            reference = gallery_device[
                torch.from_numpy(reference_indices[begin:end]).to(device)
            ]
            encoded = model.compose(reference, text_features)
        queries[begin:end] = encoded.float().cpu().numpy()
        queries.flush()
        write_json_atomic(
            progress_path,
            {"next_index": end, "query_count": len(records)},
        )

    write_json_atomic(output_dir / "image_ids.json", image_ids)
    write_json_atomic(output_dir / "pair_ids.json", [record.pair_id for record in records])
    write_json_atomic(metadata_path, expected)
    print(json.dumps(expected, ensure_ascii=False, indent=2), flush=True)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _autocast(device: torch.device, mixed_precision: str):
    enabled = device.type == "cuda" and mixed_precision in {"fp16", "bf16"}
    dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


if __name__ == "__main__":
    main()
