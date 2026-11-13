#!/usr/bin/env python3
"""Cache frozen Teacher responses and frozen Student image features for CIRR KD."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from cir_qdiag_train.distillation_config import load_distillation_config
from cir_qdiag_train.manifest import TrainManifestRecord, read_manifest
from cir_qdiag_train.model import ComposedRetrievalModel, load_checkpoint


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
CIR_PLAY_ROOT = WORKSPACE_ROOT / "CIR_PLAY"
DEFAULT_TEACHER = CIR_PLAY_ROOT / "models" / "checkpoints" / "BGE-VL-MLLM-S2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--teacher-model-name", default="BGE-VL-MLLM-S2")
    parser.add_argument("--teacher-model-path", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--teacher-batch-size", type=int, default=4)
    parser.add_argument("--student-image-batch-size", type=int, default=128)
    parser.add_argument("--skip-teacher", action="store_true")
    parser.add_argument("--skip-student", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _layout_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_image_map(path: Path, image_root: Path) -> dict[str, Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"invalid image split map: {path}")
    root = image_root.resolve()
    result: dict[str, Path] = {}
    for image_id, mapped in payload.items():
        if not isinstance(image_id, str) or not isinstance(mapped, str):
            raise TypeError(f"invalid image split entry in {path}")
        relative = Path(mapped.removeprefix("./"))
        resolved = (root / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not resolved.is_relative_to(root):
            raise ValueError(f"unsafe CIRR image path: {mapped}")
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        result[image_id] = resolved
    return result


def _candidate_ids(record: TrainManifestRecord) -> list[str]:
    distractors = sorted(
        set(record.image_set) - {record.reference_id, record.target_id}
    )
    candidates = [record.target_id, *distractors]
    if len(candidates) < 2:
        raise ValueError(f"{record.sample_id} has no group distractors")
    return candidates


def _negative_layout(
    records: Sequence[TrainManifestRecord],
    semantic_target: str,
    edit_operations: set[str],
) -> tuple[list[str], list[int], list[str]]:
    ids: list[str] = []
    owners: list[int] = []
    texts: list[str] = []
    for owner, record in enumerate(records):
        for negative in record.negatives:
            if (
                negative.semantic_target == semantic_target
                and negative.edit_operation in edit_operations
            ):
                ids.append(negative.negative_id)
                owners.append(owner)
                texts.append(negative.text)
    return ids, owners, texts


def _open_feature_cache(
    *,
    cache_dir: Path,
    ids: list[str],
    paths: list[Path],
    batch_size: int,
    encode: Callable[[Sequence[Path]], np.ndarray],
    fingerprint: str,
    force: bool,
    description: str,
) -> np.memmap:
    cache_dir.mkdir(parents=True, exist_ok=True)
    ids_path = cache_dir / "image_ids.json"
    features_path = cache_dir / "image_features.npy"
    progress_path = cache_dir / "image_progress.json"
    if force:
        for path in (ids_path, features_path, progress_path):
            path.unlink(missing_ok=True)

    start = 0
    features: np.memmap | None = None
    if ids_path.is_file() or features_path.is_file() or progress_path.is_file():
        if not (ids_path.is_file() and features_path.is_file() and progress_path.is_file()):
            raise RuntimeError(f"incomplete {description} cache; rerun with --force")
        cached_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if cached_ids != ids or progress.get("fingerprint") != fingerprint:
            raise RuntimeError(f"incompatible {description} cache; rerun with --force")
        features = np.load(features_path, mmap_mode="r+")
        if features.ndim != 2 or features.shape[0] != len(ids):
            raise ValueError(f"invalid {description} feature shape: {features.shape}")
        start = int(progress["next_index"])

    first_batch: np.ndarray | None = None
    if features is None:
        first_end = min(batch_size, len(ids))
        first_batch = np.asarray(encode(paths[:first_end]), dtype=np.float32)
        if first_batch.ndim != 2 or first_batch.shape[0] != first_end:
            raise ValueError(
                f"{description} encoder returned {first_batch.shape}, expected {first_end} rows"
            )
        features = np.lib.format.open_memmap(
            features_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(ids), first_batch.shape[1]),
        )
        features[:first_end] = _normalize(first_batch)
        features.flush()
        _write_json_atomic(ids_path, ids)
        start = first_end
        _write_json_atomic(
            progress_path,
            {
                "fingerprint": fingerprint,
                "next_index": start,
                "total": len(ids),
                "embedding_dim": int(features.shape[1]),
            },
        )

    for begin in tqdm(
        range(start, len(ids), batch_size),
        desc=description,
        initial=start // batch_size,
        total=(len(ids) + batch_size - 1) // batch_size,
    ):
        end = min(begin + batch_size, len(ids))
        encoded = np.asarray(encode(paths[begin:end]), dtype=np.float32)
        if encoded.shape != (end - begin, features.shape[1]):
            raise ValueError(f"{description} encoder returned invalid shape {encoded.shape}")
        features[begin:end] = _normalize(encoded)
        features.flush()
        _write_json_atomic(
            progress_path,
            {
                "fingerprint": fingerprint,
                "next_index": end,
                "total": len(ids),
                "embedding_dim": int(features.shape[1]),
            },
        )
    return features


def _normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or bool((norms <= 1e-12).any()):
        raise ValueError("encoder produced non-finite or zero-norm features")
    return values / norms


def _open_score_cache(path: Path, shape: tuple[int, int], force: bool) -> np.memmap:
    if force:
        path.unlink(missing_ok=True)
    if path.is_file():
        values = np.load(path, mmap_mode="r+")
        if values.shape != shape:
            raise ValueError(f"score cache shape mismatch at {path}: {values.shape} != {shape}")
        return values
    values = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
    values[:] = np.nan
    values.flush()
    return values


def prepare_teacher_cache(args: argparse.Namespace, config, records, image_map) -> None:
    output_dir = config.data.teacher_cache_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    signals_path = output_dir / "teacher_signals.npz"
    metadata_path = output_dir / "metadata.json"
    model_path = args.teacher_model_path.resolve()
    operations = set(config.data.edit_operations)
    candidate_lists = [_candidate_ids(record) for record in records]
    max_candidates = max(map(len, candidate_lists))
    candidate_ids = np.full((len(records), max_candidates), "", dtype=f"<U{max(map(len, image_map), default=1)}")
    candidate_mask = np.zeros((len(records), max_candidates), dtype=np.bool_)
    for index, values in enumerate(candidate_lists):
        candidate_ids[index, : len(values)] = values
        candidate_mask[index, : len(values)] = True
    negative_ids, negative_owners, negative_texts = _negative_layout(
        records,
        config.data.semantic_target,
        operations,
    )
    manifest_hash = _sha256(config.data.train_manifest)
    layout = {
        "schema_version": 1,
        "manifest_sha256": manifest_hash,
        "teacher_model_name": args.teacher_model_name,
        "teacher_model_path": str(model_path),
        "semantic_target": config.data.semantic_target,
        "edit_operations": sorted(operations),
        "sample_count": len(records),
        "negative_count": len(negative_ids),
        "candidate_count_histogram": {
            str(count): sum(len(values) == count for values in candidate_lists)
            for count in sorted(set(map(len, candidate_lists)))
        },
    }
    fingerprint = _layout_fingerprint(layout)
    if signals_path.is_file() and metadata_path.is_file() and not args.force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("fingerprint") != fingerprint:
            raise RuntimeError("teacher cache exists but is incompatible; rerun with --force")
        if metadata.get("complete"):
            print(f"[skip teacher] complete cache: {signals_path}", flush=True)
            return

    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    if str(CIR_PLAY_ROOT) not in sys.path:
        sys.path.insert(0, str(CIR_PLAY_ROOT))
    from run_pinpoint_bge_lincir import choose_device, make_adapter, prepare_hf_runtime_cache

    prepare_hf_runtime_cache(output_dir / "hf_runtime")
    device = choose_device(args.device)
    adapter = make_adapter(
        args.teacher_model_name,
        model_path,
        device,
        args.dtype,
        hf_cache_dir=CIR_PLAY_ROOT / "models" / "checkpoints" / "lincir" / ".cache",
        lincir_clip_model_name="large",
    )
    unique_candidate_ids = sorted({value for values in candidate_lists for value in values})
    unique_paths = [image_map[value] for value in unique_candidate_ids]
    image_features = _open_feature_cache(
        cache_dir=output_dir / "teacher_images",
        ids=unique_candidate_ids,
        paths=unique_paths,
        batch_size=args.teacher_batch_size,
        encode=lambda paths: adapter.encode_image_batch(paths),
        fingerprint=fingerprint,
        force=args.force,
        description="Teacher candidate images",
    )
    image_index = {value: index for index, value in enumerate(unique_candidate_ids)}
    candidate_indices = np.zeros((len(records), max_candidates), dtype=np.int64)
    for owner, values in enumerate(candidate_lists):
        candidate_indices[owner, : len(values)] = [image_index[value] for value in values]

    positive_scores = _open_score_cache(
        output_dir / "teacher_positive_scores.npy",
        (len(records), max_candidates),
        args.force,
    )
    negative_scores = _open_score_cache(
        output_dir / "teacher_negative_scores.npy",
        (len(negative_ids), max_candidates),
        args.force,
    )
    query_items: list[tuple[str, int, Path, str]] = []
    for owner, record in enumerate(records):
        if np.isnan(positive_scores[owner, candidate_mask[owner]]).any():
            query_items.append(("positive", owner, Path(record.reference_path), record.positive_text))
    for negative_index, (owner, text) in enumerate(zip(negative_owners, negative_texts)):
        if np.isnan(negative_scores[negative_index, candidate_mask[owner]]).any():
            query_items.append(
                ("negative", negative_index, Path(records[owner].reference_path), text)
            )

    print(f"[Teacher queries] cached={len(records) + len(negative_ids) - len(query_items)} pending={len(query_items)}")
    for batch_number, begin in enumerate(tqdm(
        range(0, len(query_items), args.teacher_batch_size),
        desc="Teacher composed queries",
    ), start=1):
        batch = query_items[begin : begin + args.teacher_batch_size]
        frame = pd.DataFrame(
            {
                "query_id": [f"kd_{begin + offset}" for offset in range(len(batch))],
                "query_image_signature": [str(item[2]) for item in batch],
                "instruction": [item[3] for item in batch],
            }
        )
        embeddings = _normalize(
            np.asarray(adapter.encode_query_batch(frame, image_cache=Path("/")), dtype=np.float32)
        )
        for embedding, (kind, item_index, _reference, _text) in zip(embeddings, batch):
            owner = item_index if kind == "positive" else negative_owners[item_index]
            valid = candidate_mask[owner]
            scores = embedding @ image_features[candidate_indices[owner, valid]].T
            if kind == "positive":
                positive_scores[item_index, valid] = scores
            else:
                negative_scores[item_index, valid] = scores
        if batch_number % 32 == 0 or begin + len(batch) == len(query_items):
            positive_scores.flush()
            negative_scores.flush()

    if np.isnan(positive_scores[candidate_mask]).any():
        raise RuntimeError("Teacher positive score cache is incomplete")
    negative_candidate_mask = candidate_mask[np.asarray(negative_owners, dtype=np.int64)]
    if np.isnan(negative_scores[negative_candidate_mask]).any():
        raise RuntimeError("Teacher negative score cache is incomplete")

    target_top1 = np.zeros(len(records), dtype=np.bool_)
    for owner in range(len(records)):
        valid_scores = positive_scores[owner, candidate_mask[owner]]
        target_top1[owner] = int(valid_scores.argmax()) == 0
    negative_owner_array = np.asarray(negative_owners, dtype=np.int64)
    teacher_target_delta = (
        positive_scores[negative_owner_array, 0] - negative_scores[:, 0]
    )
    temporary = signals_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            sample_ids=np.asarray([record.sample_id for record in records]),
            reference_ids=np.asarray([record.reference_id for record in records]),
            candidate_ids=candidate_ids,
            candidate_mask=candidate_mask,
            target_indices=np.zeros(len(records), dtype=np.int64),
            positive_scores=np.asarray(positive_scores),
            teacher_target_top1=target_top1,
            negative_ids=np.asarray(negative_ids),
            negative_owner=negative_owner_array,
            negative_scores=np.asarray(negative_scores),
        )
    temporary.replace(signals_path)
    metadata = {
        **layout,
        "fingerprint": fingerprint,
        "complete": True,
        "signals_path": str(signals_path),
        "teacher_embedding_dimension": int(image_features.shape[1]),
        "teacher_target_top1_count": int(target_top1.sum()),
        "teacher_target_top1_rate": float(target_top1.mean()),
        "semantic_target_delta_positive_count": int((teacher_target_delta > 0).sum()),
        "semantic_target_delta_positive_rate": float((teacher_target_delta > 0).mean()),
        "semantic_target_delta_mean": float(teacher_target_delta.mean()),
    }
    _write_json_atomic(metadata_path, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    adapter = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _model_from_checkpoint(checkpoint_path: Path) -> tuple[ComposedRetrievalModel, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = checkpoint.get("config", {}).get("model")
    optimizer_config = checkpoint.get("config", {}).get("optimizer", {})
    if not isinstance(model_config, dict):
        raise TypeError(f"checkpoint has no model config: {checkpoint_path}")
    model, processor = ComposedRetrievalModel.from_pretrained(
        str(model_config["pretrained_model_name_or_path"]),
        variant=str(model_config["variant"]),
        local_files_only=bool(model_config.get("local_files_only", True)),
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        hidden_dim=int(model_config["hidden_dim"]),
        dropout=float(model_config["dropout"]),
        freeze_backbone=bool(model_config["freeze_backbone"]),
        freeze_vision_encoder=bool(model_config.get("freeze_vision_encoder", False)),
        gradient_checkpointing=False,
        temperature=float(optimizer_config.get("temperature", 0.07)),
        learnable_temperature=bool(optimizer_config.get("learnable_temperature", True)),
        unfreeze_last_n_layers=model_config.get("unfreeze_last_n_layers"),
        lora_rank=model_config.get("lora_rank"),
        lora_alpha=float(model_config.get("lora_alpha", 32.0)),
        lora_dropout=float(model_config.get("lora_dropout", 0.05)),
    )
    load_checkpoint(checkpoint_path, model=model)
    return model, processor


def prepare_student_cache(args: argparse.Namespace, config, records, image_map) -> None:
    output_dir = config.data.student_image_cache_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.experiment.initialize_from.resolve()
    image_paths: dict[str, Path] = {}
    for record in records:
        image_paths[record.reference_id] = Path(record.reference_path)
        for image_id in _candidate_ids(record):
            image_paths[image_id] = image_map[image_id]
    image_ids = sorted(image_paths)
    paths = [image_paths[value] for value in image_ids]
    layout = {
        "schema_version": 1,
        "manifest_sha256": _sha256(config.data.train_manifest),
        "student_initialize_from": str(checkpoint_path),
        "student_checkpoint_size": checkpoint_path.stat().st_size,
        "image_count": len(image_ids),
    }
    fingerprint = _layout_fingerprint(layout)
    metadata_path = output_dir / "metadata.json"
    if metadata_path.is_file() and not args.force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        progress_path = output_dir / "image_progress.json"
        progress = (
            json.loads(progress_path.read_text(encoding="utf-8"))
            if progress_path.is_file()
            else {}
        )
        if (
            metadata.get("fingerprint") == fingerprint
            and metadata.get("complete")
            and progress.get("next_index") == len(image_ids)
        ):
            print(f"[skip student] complete cache: {output_dir}", flush=True)
            return

    model, processor = _model_from_checkpoint(checkpoint_path)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    def encode(batch_paths: Sequence[Path]) -> np.ndarray:
        images = []
        for path in batch_paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt")
        amp = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else torch.no_grad()
        )
        with torch.inference_mode(), amp:
            values = model.encode_image(inputs["pixel_values"].to(device))
        return values.float().cpu().numpy()

    features = _open_feature_cache(
        cache_dir=output_dir,
        ids=image_ids,
        paths=paths,
        batch_size=args.student_image_batch_size,
        encode=encode,
        fingerprint=fingerprint,
        force=args.force,
        description="Student frozen images",
    )
    progress = json.loads((output_dir / "image_progress.json").read_text(encoding="utf-8"))
    metadata = {
        **layout,
        "fingerprint": fingerprint,
        "complete": int(progress["next_index"]) == len(image_ids),
        "embedding_dimension": int(features.shape[1]),
        "features_path": str((output_dir / "image_features.npy").resolve()),
    }
    _write_json_atomic(metadata_path, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    if args.teacher_batch_size <= 0 or args.student_image_batch_size <= 0:
        raise ValueError("cache batch sizes must be positive")
    started = time.perf_counter()
    config = load_distillation_config(args.config)
    records = read_manifest(config.data.train_manifest)
    image_map = _load_image_map(config.data.train_image_split_path, config.data.image_root)
    if not args.skip_teacher:
        prepare_teacher_cache(args, config, records, image_map)
    if not args.skip_student:
        prepare_student_cache(args, config, records, image_map)
    print(f"[done] cache preparation elapsed_seconds={time.perf_counter() - started:.1f}")


if __name__ == "__main__":
    main()
