#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from cir_qdiag_train.on_policy_config import (
    OnPolicyDistillationConfig,
    load_on_policy_config,
)
from cir_qdiag_train.on_policy_trainer import train_on_policy_distillation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--teacher-endpoint-url")
    parser.add_argument("--teacher-model-name")
    parser.add_argument("--teacher-api-format", choices=("rerank", "score"))
    parser.add_argument(
        "--teacher-image-transport", choices=("data_url", "file_url")
    )
    parser.add_argument("--teacher-max-concurrent-requests", type=int)
    parser.add_argument("--live-selection-image-batch-size", type=int)
    parser.add_argument("--mining-image-batch-size", type=int)
    parser.add_argument("--pixel-cache-build-batch-size", type=int)
    parser.add_argument("--pixel-cache-workers", type=int)
    parser.add_argument("--pixel-cache-build-prefetch-factor", type=int)
    parser.add_argument("--pixel-cache-transfer-prefetch-batches", type=int)
    parser.add_argument("--pixel-cache-dtype", choices=("float16", "float32"))
    parser.add_argument("--batch-size-per-gpu", type=int)
    parser.add_argument("--num-workers", type=int)
    args = parser.parse_args()
    config = load_on_policy_config(args.config)
    teacher_updates = {
        key: value
        for key, value in {
            "endpoint_url": args.teacher_endpoint_url,
            "model_name": args.teacher_model_name,
            "api_format": args.teacher_api_format,
            "image_transport": args.teacher_image_transport,
            "max_concurrent_requests": args.teacher_max_concurrent_requests,
        }.items()
        if value is not None
    }
    runtime_updates = {
        "mining": {
            key: value
            for key, value in {
                "live_selection_image_batch_size": (
                    args.live_selection_image_batch_size
                ),
                "image_batch_size": args.mining_image_batch_size,
                "pixel_cache_build_batch_size": (
                    args.pixel_cache_build_batch_size
                ),
                "pixel_cache_workers": args.pixel_cache_workers,
                "pixel_cache_build_prefetch_factor": (
                    args.pixel_cache_build_prefetch_factor
                ),
                "pixel_cache_transfer_prefetch_batches": (
                    args.pixel_cache_transfer_prefetch_batches
                ),
                "pixel_cache_dtype": args.pixel_cache_dtype,
            }.items()
            if value is not None
        },
        "optimizer": {
            "batch_size_per_gpu": args.batch_size_per_gpu
        }
        if args.batch_size_per_gpu is not None
        else {},
        "data": {"num_workers": args.num_workers}
        if args.num_workers is not None
        else {},
    }
    if teacher_updates or any(runtime_updates.values()):
        payload = config.model_dump()
        payload["teacher"].update(teacher_updates)
        for section, updates in runtime_updates.items():
            payload[section].update(updates)
        config = OnPolicyDistillationConfig.model_validate(payload)
    train_on_policy_distillation(
        config,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
