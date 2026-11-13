#!/usr/bin/env python3
"""Evaluate a project CLIP composed-retrieval checkpoint on FashionIQ val."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml

from cir_qdiag_train.config import load_config
from cir_qdiag_train.fashioniq_evaluation import (
    FASHIONIQ_CATEGORIES,
    aggregate_category_metrics,
    cache_feature_matrix,
    evaluate_feature_files,
    load_fashioniq_val,
)
from cir_qdiag_train.inference_data import (
    build_inference_loader,
    inference_loader_settings,
    inference_tensor_to_device,
)
from cir_qdiag_train.model import ComposedRetrievalModel, load_checkpoint
from cir_qdiag_train.on_policy_config import load_on_policy_config
from cir_qdiag_train.on_policy_mining import write_json_atomic

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT.parents[1] / "CIR_new/datasets/fashion-iq"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-batch-size", type=int, default=256)
    parser.add_argument("--query-batch-size", type=int, default=256)
    parser.add_argument("--score-batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--force-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in ("image_batch_size", "query_batch_size", "score_batch_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    config = _load_model_config(args.config)
    workers, prefetch = inference_loader_settings(
        default_workers=config.evaluation.num_workers,
        default_prefetch=config.evaluation.prefetch_factor,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    checkpoint = args.checkpoint.resolve() if args.checkpoint is not None else None
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema_version": 1,
        "model_name": args.model_name,
        "config": str(args.config.resolve()),
        "checkpoint": "pretrained" if checkpoint is None else str(checkpoint),
        "checkpoint_size": None if checkpoint is None else checkpoint.stat().st_size,
        "checkpoint_mtime_ns": (
            None if checkpoint is None else checkpoint.stat().st_mtime_ns
        ),
        "dataset_root": str(args.dataset_root.resolve()),
        "device": args.device,
    }
    _prepare_output(output_dir, identity, force=args.force_cache)
    final_path = output_dir / "metrics.json"
    if final_path.is_file() and not args.force_cache:
        print(final_path.read_text(encoding="utf-8"), flush=True)
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
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
        gradient_checkpointing=False,
        temperature=config.optimizer.temperature,
        learnable_temperature=config.optimizer.learnable_temperature,
        unfreeze_last_n_layers=config.model.unfreeze_last_n_layers,
        lora_rank=config.model.lora_rank,
        lora_alpha=config.model.lora_alpha,
        lora_dropout=config.model.lora_dropout,
    )
    if checkpoint is not None:
        load_checkpoint(checkpoint, model=model)
    model.to(device).eval()

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
            encode_batches=lambda start, paths=gallery_paths: _iter_encoded_batches(
                model=model,
                processor=processor,
                image_paths=paths,
                texts=None,
                batch_size=args.image_batch_size,
                num_workers=workers,
                prefetch_factor=prefetch,
                pin_memory=config.evaluation.pin_memory,
                start=start,
                device=device,
                mixed_precision=config.optimizer.mixed_precision,
            ),
            description=f"{args.model_name} FashionIQ-{category} gallery",
        )
        cache_feature_matrix(
            query_path,
            category_dir / "query_progress.json",
            count=len(records),
            batch_size=args.query_batch_size,
            encode_batches=lambda start, rows=records: _iter_encoded_batches(
                model=model,
                processor=processor,
                image_paths=[
                    args.dataset_root.resolve() / "images" / f"{row.reference_id}.png"
                    for row in rows
                ],
                texts=[row.text for row in rows],
                batch_size=args.query_batch_size,
                num_workers=workers,
                prefetch_factor=prefetch,
                pin_memory=config.evaluation.pin_memory,
                start=start,
                device=device,
                mixed_precision=config.optimizer.mixed_precision,
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

    source = "pretrained" if checkpoint is None else str(checkpoint)
    result = aggregate_category_metrics(
        model_name=args.model_name,
        model_source=source,
        category_metrics=category_metrics,
    )
    write_json_atomic(final_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def _load_model_config(path: Path):
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "objective" in payload and "mining" in payload:
        return load_on_policy_config(path)
    return load_config(path)


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


def _iter_encoded_batches(
    *, model, processor, image_paths, texts, batch_size, num_workers,
    prefetch_factor, pin_memory, start, device, mixed_precision,
):
    loader = build_inference_loader(
        image_paths=image_paths,
        texts=texts,
        processor=processor,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory and device.type == "cuda",
        start=start,
    )
    for inputs in loader:
        yield _encode_prepared_batch(model, inputs, device, mixed_precision)


@torch.inference_mode()
def _encode_prepared_batch(
    model,
    inputs,
    device: torch.device,
    mixed_precision: str,
) -> np.ndarray:
    with _autocast(device, mixed_precision):
        pixels = inference_tensor_to_device(inputs["pixel_values"], device)
        if "input_ids" not in inputs:
            return model.encode_image(pixels).float().cpu().numpy()
        reference_features = model.encode_reference(pixels)
        text_features = model.encode_text(
            inference_tensor_to_device(inputs["input_ids"], device),
            None
            if inputs.get("attention_mask") is None
            else inference_tensor_to_device(inputs["attention_mask"], device),
        )
        features = model.compose(reference_features, text_features)
    return features.float().cpu().numpy()


def _autocast(device: torch.device, mixed_precision: str):
    if device.type != "cuda" or mixed_precision == "no":
        return nullcontext()
    dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


if __name__ == "__main__":
    main()
