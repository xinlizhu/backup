#!/usr/bin/env python3
"""Cache the immutable image side of a Qwen query-LoRA OPD Student."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from cir_qdiag_train.on_policy_data import load_image_map
from cir_qdiag_train.on_policy_mining import write_json_atomic
from cir_qdiag_train.qwen_on_policy_config import load_qwen_on_policy_config
from cir_qdiag_train.qwen_student import QwenQueryLoRAStudent
from prepare_on_policy_teacher_cache import _open_resumable_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-batch-size", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.image_batch_size <= 0:
        raise ValueError("--image-batch-size must be positive")

    config = load_qwen_on_policy_config(args.config)
    if config.model.train_gallery:
        raise ValueError("Shared LoRA has no immutable gallery: training refreshes it automatically")
    image_ids, image_paths = load_image_map(
        config.data.train_image_split_path, config.data.image_root
    )
    output_dir = config.data.student_gallery_cache_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    expected = {
        "schema_version": 1,
        "student_type": config.model.variant,
        "base_model_path": str(
            Path(config.model.pretrained_model_name_or_path).resolve()
        ),
        "gallery_count": len(image_ids),
        "feature_dtype": config.mining.cache_dtype,
        "image_encoder": ("original Qwen3-VL-2B-Instruct with query LoRA disabled"
                          if config.model.variant == "qwen3_vl_instruct_query_lora" else
                          "original Qwen3-VL-Embedding with query LoRA disabled"),
    }
    if config.model.variant == "qwen3_vl_instruct_query_lora":
        expected["preprocessing"] = {
            "max_length": config.model.max_length,
            "min_pixels": config.model.min_pixels,
            "max_pixels": config.model.max_pixels,
            "instruction": QwenQueryLoRAStudent.TARGET_INSTRUCTION,
        }
        if not args.force:
            existing = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
            if existing and any(existing.get(k) != v for k, v in expected.items()):
                raise ValueError("Instruct gallery metadata mismatch; use a separate cache directory")
            if not existing and (output_dir / "gallery_features.npy").exists():
                raise ValueError("Instruct gallery lacks provenance; use a separate cache directory")
            # Persist provenance before encoding so partial cache resume is safe.
            if not existing:
                write_json_atomic(metadata_path, {**expected, "complete": False})
    if metadata_path.is_file() and not args.force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (config.model.variant == "qwen3_vl_instruct_query_lora"
                and metadata.get("complete")
                and not all((output_dir / name).is_file() for name in ("gallery_features.npy", "image_ids.json"))):
            raise ValueError("Instruct gallery marked complete but files are missing")
        if metadata.get("complete") and all(
            metadata.get(key) == value for key, value in expected.items()
        ):
            print(f"[skip] complete Qwen Student gallery: {output_dir}", flush=True)
            return
    if args.force:
        for name in (
            "gallery_features.npy",
            "gallery_progress.json",
            "image_ids.json",
            "metadata.json",
        ):
            (output_dir / name).unlink(missing_ok=True)

    if config.model.variant == "qwen3_vl_instruct_query_lora" and not metadata_path.is_file():
        write_json_atomic(metadata_path, {**expected, "complete": False})

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    student = QwenQueryLoRAStudent(
        config.model,
        device=device,
        mixed_precision=config.optimizer.mixed_precision,
    )
    student.eval()
    started = time.perf_counter()
    np_dtype = np.float16 if config.mining.cache_dtype == "float16" else np.float32
    gallery = _open_resumable_features(
        path=output_dir / "gallery_features.npy",
        progress_path=output_dir / "gallery_progress.json",
        count=len(image_ids),
        batch_size=args.image_batch_size,
        np_dtype=np_dtype,
        encode=lambda begin, end: student.encode_prepared_images(
            student.prepare_image_batch(image_paths[begin:end])
        )
        .cpu()
        .numpy(),
        description="Qwen-2B immutable Student train gallery",
    )
    write_json_atomic(output_dir / "image_ids.json", image_ids)
    write_json_atomic(
        metadata_path,
        {
            **expected,
            "complete": True,
            "embedding_dimension": int(gallery.shape[1]),
            "elapsed_seconds_this_process": time.perf_counter() - started,
        },
    )
    print(metadata_path.read_text(encoding="utf-8"), flush=True)
    student = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
