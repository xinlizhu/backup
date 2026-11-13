from __future__ import annotations

import json
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from cir_qdiag_train.model import ComposedRetrievalModel
from cir_qdiag_train.inference_data import (
    build_inference_loader,
    inference_loader_settings,
    inference_tensor_to_device,
)
from cir_qdiag_train.http_reranker import HttpRerankerClient
from cir_qdiag_train.on_policy_config import OnPolicyDistillationConfig
from cir_qdiag_train.on_policy_data import (
    CIRRPositiveRecord,
    NegativeQueryLayout,
    load_negative_query_layout,
)
from cir_qdiag_train.pixel_cache import (
    GalleryPixelCache,
    iter_prefetched_pixel_chunks,
)


@dataclass
class InMemoryStudentReservoir:
    """Ephemeral epoch-start candidates and anchor scores.

    The arrays intentionally remain in host memory and are never written to the
    experiment directory. Candidate zero is always the gold target; the
    remaining columns are the epoch-start Student's highest-scoring non-targets.
    """

    candidate_indices: np.ndarray
    anchor_scores: np.ndarray | None
    target_indices: np.ndarray
    raw_target_in_reservoir: np.ndarray
    raw_target_in_candidate_topk: np.ndarray
    metadata: dict[str, Any]
    teacher_scores: np.ndarray | None = None


@torch.inference_mode()
def build_student_reservoir_in_memory(
    *,
    config: OnPolicyDistillationConfig,
    model: ComposedRetrievalModel,
    processor,
    records: list[CIRRPositiveRecord],
    image_ids: list[str],
    image_paths: list[Path],
    epoch: int,
    global_step: int,
    device: torch.device,
    pixel_cache: GalleryPixelCache | None = None,
) -> InMemoryStudentReservoir:
    """Mine an epoch-start Student Top-M reservoir without persistent caches."""

    reservoir_top_k = config.mining.reservoir_top_k
    if not config.mining.live_reservoir_selection or reservoir_top_k is None:
        raise ValueError("in-memory reservoir mining is not enabled")
    if config.mining.persist_candidate_cache:
        raise ValueError("in-memory reservoir mining cannot persist candidate caches")
    if reservoir_top_k >= len(image_ids):
        raise ValueError("reservoir Top-K must be smaller than the train gallery")

    started = time.perf_counter()
    model.eval()
    image_index = {image_id: index for index, image_id in enumerate(image_ids)}
    reference_indices = np.asarray(
        [image_index[record.reference_id] for record in records], dtype=np.int64
    )
    target_indices = np.asarray(
        [image_index[record.target_id] for record in records], dtype=np.int64
    )
    reference_parts = []
    gallery = _encode_student_gallery(
        model=model,
        processor=processor,
        image_paths=image_paths,
        batch_size=config.mining.image_batch_size,
        device=device,
        mixed_precision=config.optimizer.mixed_precision,
        pixel_cache=pixel_cache,
        transfer_prefetch_batches=(
            config.mining.pixel_cache_transfer_prefetch_batches
        ),
        pin_memory=config.mining.pixel_cache_pin_memory,
        num_workers=config.data.num_workers,
        prefetch_factor=config.data.prefetch_factor,
        reference_features_out=reference_parts,
    )
    gallery_device = gallery.to(device=device, dtype=torch.float32)
    reference_device = torch.cat(reference_parts).to(device=device, dtype=torch.float32)
    del gallery

    sample_count = len(records)
    candidate_indices = np.empty(
        (sample_count, reservoir_top_k), dtype=np.int32
    )
    anchor_dtype = (
        np.float16 if config.mining.cache_dtype == "float16" else np.float32
    )
    anchor_scores = (
        np.empty((sample_count, reservoir_top_k), dtype=anchor_dtype)
        if config.objective.anchor_kl_enabled
        else None
    )
    raw_target_in_reservoir = np.zeros(sample_count, dtype=np.bool_)
    raw_target_in_candidate_topk = np.zeros(sample_count, dtype=np.bool_)
    candidate_top_k = config.mining.top_k
    texts = [record.text for record in records]

    for begin in tqdm(
        range(0, sample_count, config.mining.query_batch_size),
        desc=f"Anchor Top-{reservoir_top_k} in-memory reservoir epoch {epoch}",
    ):
        end = min(begin + config.mining.query_batch_size, sample_count)
        text_inputs = processor(
            text=texts[begin:end],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with _autocast(device, config.optimizer.mixed_precision):
            text_features = model.encode_text(
                text_inputs["input_ids"].to(device),
                (
                    None
                    if text_inputs.get("attention_mask") is None
                    else text_inputs["attention_mask"].to(device)
                ),
            )
            reference_tensor = torch.from_numpy(
                reference_indices[begin:end]
            ).to(device)
            queries = model.compose(
                reference_device[reference_tensor],
                text_features,
            )
        all_scores = queries.float() @ gallery_device.t()
        all_scores.scatter_(1, reference_tensor[:, None], -torch.inf)
        raw_scores, raw_indices = torch.topk(
            all_scores, k=reservoir_top_k, dim=1
        )
        local_targets = torch.from_numpy(target_indices[begin:end]).to(device)
        target_scores = all_scores.gather(1, local_targets[:, None]).squeeze(1)
        raw_indices_np = raw_indices.cpu().numpy()
        raw_scores_np = raw_scores.cpu().numpy()
        target_scores_np = target_scores.cpu().numpy()

        for offset, target_index in enumerate(target_indices[begin:end]):
            row = begin + offset
            raw_row = raw_indices_np[offset]
            raw_value_row = raw_scores_np[offset]
            target_mask = raw_row == target_index
            target_positions = np.flatnonzero(target_mask)
            raw_target_in_reservoir[row] = bool(target_positions.size)
            raw_target_in_candidate_topk[row] = bool(
                (raw_row[:candidate_top_k] == target_index).any()
            )
            non_target_mask = ~target_mask
            non_target_indices = raw_row[non_target_mask]
            non_target_scores = raw_value_row[non_target_mask]
            if len(non_target_indices) < reservoir_top_k - 1:
                raise RuntimeError("anchor reservoir has too few non-target images")
            candidate_indices[row, 0] = int(target_index)
            candidate_indices[row, 1:] = non_target_indices[
                : reservoir_top_k - 1
            ].astype(np.int32, copy=False)
            if anchor_scores is not None:
                anchor_scores[row, 0] = target_scores_np[offset]
                anchor_scores[row, 1:] = non_target_scores[
                    : reservoir_top_k - 1
                ].astype(anchor_dtype, copy=False)

        del all_scores, raw_scores, raw_indices, queries, text_features

    del gallery_device, reference_device
    teacher_scores = None
    if config.objective.needs_teacher_scores:
        if config.teacher.backend != "embedding_cache":
            raise ValueError("in-memory Teacher scores require embedding_cache")
        teacher_scores, _ = _score_teacher_candidates(
            config=config,
            candidate_indices=candidate_indices,
            records=records,
            image_ids=image_ids,
            image_paths=image_paths,
            output_dir=config.experiment.output_dir,
            device=device,
        )
    metadata = {
        "schema_version": 1,
        "complete": True,
        "ephemeral_in_memory": True,
        "epoch": epoch,
        "source_global_step": global_step,
        "sample_count": sample_count,
        "gallery_count": len(image_ids),
        "top_k": candidate_top_k,
        "reservoir_top_k": reservoir_top_k,
        "candidate_source": "student",
        "refresh_mode": "every_epoch",
        "loss_type": config.objective.loss_type,
        "target_forced_into_reservoir": True,
        "target_forced_into_live_candidates": True,
        "raw_target_recall_at_k": float(raw_target_in_candidate_topk.mean()),
        "raw_target_recall_at_reservoir_k": float(
            raw_target_in_reservoir.mean()
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return InMemoryStudentReservoir(
        candidate_indices=candidate_indices,
        anchor_scores=anchor_scores,
        target_indices=target_indices.astype(np.int32, copy=False),
        raw_target_in_reservoir=raw_target_in_reservoir,
        raw_target_in_candidate_topk=raw_target_in_candidate_topk,
        metadata=metadata,
        teacher_scores=teacher_scores,
    )


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@torch.inference_mode()
def refresh_on_policy_candidates(
    *,
    config: OnPolicyDistillationConfig,
    model: ComposedRetrievalModel,
    processor,
    records: list[CIRRPositiveRecord],
    image_ids: list[str],
    image_paths: list[Path],
    epoch: int,
    global_step: int,
    device: torch.device,
) -> tuple[Path, dict[str, Any]]:
    """Select candidates, force the target, and optionally score with Teacher."""

    output_dir = config.experiment.output_dir / "on_policy" / f"epoch_{epoch:03d}"
    metadata_path = output_dir / "metadata.json"
    teacher_identity = _teacher_identity(config)
    uses_mixed_candidates = config.mining.include_negative_queries_as_candidates
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        required = [
            output_dir / "candidate_indices.npy",
            output_dir / "target_indices.npy",
            output_dir / "raw_target_in_topk.npy",
        ]
        if config.mining.candidate_source == "student":
            required.append(output_dir / "gallery_features.npy")
        if uses_mixed_candidates:
            required.extend(
                [
                    output_dir / "retrieved_image_candidate_indices.npy",
                    output_dir / "negative_query_candidate_indices.npy",
                ]
            )
        if config.objective.needs_teacher_scores:
            required.extend(
                [
                    output_dir / "teacher_scores.npy",
                    output_dir / "teacher_target_top1.npy",
                ]
            )
        if not config.mining.force_target_insertion:
            required.append(output_dir / "teacher_filter_target_ranks.npy")
        if (
            metadata.get("complete")
            and int(metadata.get("epoch", -1)) == epoch
            and int(metadata.get("source_global_step", -1)) == global_step
            and int(metadata.get("top_k", -1)) == config.mining.top_k
            and metadata.get("candidate_source") == config.mining.candidate_source
            and metadata.get("loss_type") == config.objective.loss_type
            and metadata.get("target_forced_into_candidates", True) == config.mining.force_target_insertion
            and bool(metadata.get("mixed_negative_query_candidates", False))
            == uses_mixed_candidates
            and (
                config.teacher.backend == "embedding_cache"
                or metadata.get("teacher_identity") == teacher_identity
            )
            and all(path.is_file() for path in required)
        ):
            print(f"[reuse mining] {output_dir}", flush=True)
            return output_dir, metadata

    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    image_index = {image_id: index for index, image_id in enumerate(image_ids)}
    reference_indices = np.asarray(
        [image_index[record.reference_id] for record in records], dtype=np.int64
    )
    target_indices = np.asarray(
        [image_index[record.target_id] for record in records], dtype=np.int64
    )
    negative_layout = None
    if uses_mixed_candidates:
        if config.data.negative_query_manifest is None:
            raise ValueError("mixed q- candidates require a negative-query manifest")
        negative_layout = load_negative_query_layout(
            config.data.negative_query_manifest,
            records,
        )
    cache_dtype = np.float16 if config.mining.cache_dtype == "float16" else np.float32
    top_k = config.mining.top_k
    if top_k >= len(image_ids):
        raise ValueError("mining Top-K must be smaller than the CIRR train gallery")
    if config.mining.candidate_source == "student":
        reference_parts = []
        gallery = _encode_student_gallery(
            model=model,
            processor=processor,
            image_paths=image_paths,
            batch_size=config.mining.image_batch_size,
            device=device,
            mixed_precision=config.optimizer.mixed_precision,
            num_workers=config.data.num_workers,
            prefetch_factor=config.data.prefetch_factor,
            reference_features_out=reference_parts,
        )
        _save_array_atomic(
            output_dir / "gallery_features.npy",
            gallery.numpy().astype(cache_dtype, copy=False),
        )
        retrieved_candidate_indices, raw_target_in_topk = _mine_student_topk(
            config=config,
            model=model,
            processor=processor,
            records=records,
            gallery=gallery,
            reference_gallery=torch.cat(reference_parts),
            reference_indices=reference_indices,
            target_indices=target_indices,
            epoch=epoch,
            device=device,
        )
        del gallery
    elif config.mining.candidate_source == "teacher":
        retrieved_candidate_indices, raw_target_in_topk = _mine_teacher_topk(
            config=config,
            records=records,
            image_ids=image_ids,
            reference_indices=reference_indices,
            target_indices=target_indices,
            epoch=epoch,
            device=device,
        )
    else:
        retrieved_candidate_indices, raw_target_in_topk = _sample_random_candidates(
            sample_count=len(records),
            gallery_count=len(image_ids),
            reference_indices=reference_indices,
            target_indices=target_indices,
            top_k=top_k,
            seed=config.experiment.seed + epoch,
        )

    if config.mining.force_target_insertion and not np.array_equal(retrieved_candidate_indices[:, 0], target_indices):
        raise RuntimeError("forced on-policy candidates lost a ground-truth target")
    negative_query_candidate_indices = None
    candidate_indices = retrieved_candidate_indices
    if negative_layout is not None:
        candidate_indices, negative_query_candidate_indices = (
            build_mixed_candidate_layout(
                retrieved_candidate_indices,
                negative_layout,
                top_k=top_k,
                max_negative_queries=(
                    config.mining.max_negative_query_candidates_per_sample
                ),
            )
        )
        _save_array_atomic(
            output_dir / "retrieved_image_candidate_indices.npy",
            retrieved_candidate_indices,
        )
        _save_array_atomic(
            output_dir / "negative_query_candidate_indices.npy",
            negative_query_candidate_indices,
        )
    _save_array_atomic(output_dir / "candidate_indices.npy", candidate_indices)
    _save_array_atomic(output_dir / "target_indices.npy", target_indices)
    _save_array_atomic(output_dir / "raw_target_in_topk.npy", raw_target_in_topk)
    teacher_metadata: dict[str, Any] = {}
    if config.objective.needs_teacher_scores:
        teacher_scores, teacher_target_top1 = _score_teacher_candidates(
            config=config,
            candidate_indices=retrieved_candidate_indices,
            records=records,
            image_ids=image_ids,
            image_paths=image_paths,
            output_dir=output_dir,
            device=device,
        )
        if (
            negative_layout is not None
            and negative_query_candidate_indices is not None
        ):
            teacher_scores = mix_teacher_candidate_scores(
                image_scores=teacher_scores,
                candidate_image_indices=candidate_indices,
                negative_query_candidate_indices=(
                    negative_query_candidate_indices
                ),
                negative_query_scores=_load_teacher_negative_query_scores(
                    config,
                    negative_layout,
                ),
            )
            teacher_target_top1 = teacher_scores.argmax(axis=1) == 0
        filter_scores = teacher_scores
        if not config.mining.force_target_insertion:
            reference_candidates = np.stack([
                force_target_into_topk(row, int(target), top_k)[0]
                for row, target in zip(candidate_indices, target_indices, strict=True)
            ])
            filter_scores, _ = _score_teacher_candidates(
                config=config, candidate_indices=reference_candidates,
                records=records, image_ids=image_ids, image_paths=image_paths,
                output_dir=output_dir, device=device,
            )
            # Match the original loss, which computes eligibility from cached scores.
            filter_scores = filter_scores.astype(cache_dtype).astype(np.float32)
            teacher_target_top1 = filter_scores.argmax(axis=1) == 0
            filter_ranks = 1 + (filter_scores[:, 1:] > filter_scores[:, :1]).sum(axis=1)
            _save_array_atomic(output_dir / "teacher_filter_target_ranks.npy", filter_ranks)
        _save_array_atomic(
            output_dir / "teacher_scores.npy",
            teacher_scores.astype(cache_dtype, copy=False),
        )
        _save_array_atomic(
            output_dir / "teacher_target_top1.npy", teacher_target_top1
        )
        teacher_target_ranks = 1 + (
            filter_scores[:, 1:] > filter_scores[:, :1]
        ).sum(axis=1)
        teacher_metadata = {
            "teacher_backend": config.teacher.backend,
            "teacher_identity": teacher_identity,
            "teacher_target_top1_count": int(teacher_target_top1.sum()),
            "teacher_target_top1_rate": float(teacher_target_top1.mean()),
            "teacher_target_top10_count": int((teacher_target_ranks <= 10).sum()),
            "teacher_target_top10_rate": float((teacher_target_ranks <= 10).mean()),
            "teacher_target_mean_rank": float(teacher_target_ranks.mean()),
            "teacher_score_cache_dtype": config.mining.cache_dtype,
            **_teacher_score_statistics(
                teacher_scores,
                temperature=config.objective.rank_temperature,
            ),
        }
    metadata = {
        "schema_version": 1,
        "complete": True,
        "epoch": epoch,
        "source_global_step": global_step,
        "sample_count": len(records),
        "gallery_count": len(image_ids),
        "top_k": top_k,
        "candidate_source": config.mining.candidate_source,
        "refresh_mode": config.mining.refresh_mode,
        "loss_type": config.objective.loss_type,
        "teacher_filter_applied": config.objective.uses_teacher_filter,
        "candidate_layout": (
            "natural_student_top_k"
            if not config.mining.force_target_insertion else
            "target_at_0_plus_qdiag_qminus_plus_top_non_target_images"
            if negative_layout is not None
            else "target_at_0_plus_random_non_target_images"
            if config.mining.candidate_source == "random"
            else "target_at_0_plus_top_(k-1)_non_target"
        ),
        "target_forced_into_candidates": config.mining.force_target_insertion,
        "teacher_filter_support": (
            "target_plus_student_top_k_minus_1_non_targets"
            if not config.mining.force_target_insertion else "loss_candidates"
        ),
        "mixed_negative_query_candidates": negative_layout is not None,
        "available_negative_query_count": (
            0 if negative_layout is None else len(negative_layout.ids)
        ),
        "inserted_negative_query_candidate_count": (
            0
            if negative_query_candidate_indices is None
            else int((negative_query_candidate_indices >= 0).sum())
        ),
        "truncated_negative_query_candidate_count": (
            0
            if negative_layout is None or negative_query_candidate_indices is None
            else len(negative_layout.ids)
            - int((negative_query_candidate_indices >= 0).sum())
        ),
        "samples_with_negative_query_candidates": (
            0
            if negative_layout is None
            else sum(bool(values) for values in negative_layout.indices_by_owner)
        ),
        "mean_negative_query_candidates_per_sample": (
            0.0
            if negative_layout is None
            else float((negative_query_candidate_indices >= 0).sum(axis=1).mean())
        ),
        "max_negative_query_candidates_per_sample": (
            0
            if negative_layout is None
            else int((negative_query_candidate_indices >= 0).sum(axis=1).max())
        ),
        "raw_target_recall_at_k": float(raw_target_in_topk.mean()),
        **teacher_metadata,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json_atomic(metadata_path, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output_dir, metadata


def build_mixed_candidate_layout(
    retrieved_image_indices: np.ndarray,
    negative_layout: NegativeQueryLayout,
    *,
    top_k: int,
    max_negative_queries: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace leading hard-image slots with owned composed q- candidates."""

    retrieved = np.asarray(retrieved_image_indices, dtype=np.int64)
    if retrieved.ndim != 2 or retrieved.shape[1] != top_k:
        raise ValueError("retrieved image candidates have an invalid shape")
    if retrieved.shape[0] != len(negative_layout.indices_by_owner):
        raise ValueError("q- owners do not match retrieved candidate rows")
    image_candidates = np.full(retrieved.shape, -1, dtype=np.int64)
    negative_candidates = np.full(retrieved.shape, -1, dtype=np.int64)
    for owner, negative_indices in enumerate(negative_layout.indices_by_owner):
        if max_negative_queries is not None:
            negative_indices = negative_indices[:max_negative_queries]
        negative_count = len(negative_indices)
        if negative_count > top_k - 2:
            raise ValueError(
                f"pair row {owner} has {negative_count} q- candidates, but Top-{top_k} "
                "must retain the target and at least one hard image"
            )
        image_candidates[owner, 0] = retrieved[owner, 0]
        if negative_count:
            negative_candidates[owner, 1 : 1 + negative_count] = negative_indices
        image_candidates[owner, 1 + negative_count :] = retrieved[
            owner, 1 : top_k - negative_count
        ]
    occupied = (image_candidates >= 0) ^ (negative_candidates >= 0)
    if not bool(occupied.all()):
        raise RuntimeError("mixed candidate layout contains an empty or ambiguous slot")
    return image_candidates, negative_candidates


def mix_teacher_candidate_scores(
    *,
    image_scores: np.ndarray,
    candidate_image_indices: np.ndarray,
    negative_query_candidate_indices: np.ndarray,
    negative_query_scores: np.ndarray,
) -> np.ndarray:
    """Build q+ scores over a fixed-width image/q- mixed candidate set."""

    image_values = np.asarray(image_scores, dtype=np.float32)
    image_indices = np.asarray(candidate_image_indices, dtype=np.int64)
    negative_indices = np.asarray(
        negative_query_candidate_indices, dtype=np.int64
    )
    if image_values.shape != image_indices.shape or image_indices.shape != negative_indices.shape:
        raise ValueError("mixed Teacher candidate arrays have inconsistent shapes")
    result = np.empty_like(image_values)
    for owner in range(len(result)):
        image_mask = image_indices[owner] >= 0
        negative_mask = negative_indices[owner] >= 0
        if not bool((image_mask ^ negative_mask).all()):
            raise ValueError("each mixed candidate must be exactly one embedding type")
        image_count = int(image_mask.sum())
        result[owner, image_mask] = image_values[owner, :image_count]
        result[owner, negative_mask] = negative_query_scores[
            negative_indices[owner, negative_mask]
        ]
    return result


def _encode_student_gallery(
    *,
    model: ComposedRetrievalModel,
    processor,
    image_paths: list[Path],
    batch_size: int,
    device: torch.device,
    mixed_precision: str,
    pixel_cache: GalleryPixelCache | None = None,
    transfer_prefetch_batches: int = 2,
    pin_memory: bool = True,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    reference_features_out: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    features: list[torch.Tensor] = []
    if pixel_cache is not None:
        if len(pixel_cache) != len(image_paths):
            raise ValueError("gallery pixel cache does not match the image split")
        pixel_batches = iter_prefetched_pixel_chunks(
            cache=pixel_cache,
            indices=torch.arange(len(image_paths), dtype=torch.long),
            batch_size=batch_size,
            prefetch_batches=transfer_prefetch_batches,
            pin_memory=pin_memory and device.type == "cuda",
        )
        for cached_pixels in tqdm(
            pixel_batches,
            total=(len(image_paths) + batch_size - 1) // batch_size,
            desc="Student train gallery from RAM",
        ):
            device_pixels = cached_pixels.to(
                device,
                non_blocking=cached_pixels.is_pinned(),
            )
            with _autocast(device, mixed_precision):
                if reference_features_out is None:
                    encoded = model.encode_image(device_pixels)
                else:
                    raw = model._encode_image_raw(device_pixels)
                    reference_features_out.append(model.reference_from_raw(raw).float().cpu())
                    encoded = model.target_from_raw(raw)
            features.append(encoded.float().cpu())
            del cached_pixels, device_pixels, encoded
    else:
        workers, prefetch = inference_loader_settings(
            default_workers=num_workers, default_prefetch=prefetch_factor,
        )
        loader = build_inference_loader(
            image_paths=image_paths,
            processor=processor,
            batch_size=batch_size,
            num_workers=workers,
            prefetch_factor=prefetch,
            pin_memory=pin_memory and device.type == "cuda",
        )
        for inputs in tqdm(loader, desc="Student train gallery"):
            with _autocast(device, mixed_precision):
                pixels = inference_tensor_to_device(inputs["pixel_values"], device)
                if reference_features_out is None:
                    encoded = model.encode_image(pixels)
                else:
                    raw = model._encode_image_raw(pixels)
                    reference_features_out.append(model.reference_from_raw(raw).float().cpu())
                    encoded = model.target_from_raw(raw)
            features.append(encoded.float().cpu())
    gallery = torch.cat(features, dim=0)
    if not bool(torch.isfinite(gallery).all()):
        raise ValueError("Student gallery contains non-finite features")
    return torch.nn.functional.normalize(gallery, dim=1)


@torch.inference_mode()
def _mine_student_topk(
    *,
    config: OnPolicyDistillationConfig,
    model: ComposedRetrievalModel,
    processor,
    records: list[CIRRPositiveRecord],
    gallery: torch.Tensor,
    reference_gallery: torch.Tensor | None = None,
    reference_indices: np.ndarray,
    target_indices: np.ndarray,
    epoch: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    top_k = config.mining.top_k
    gallery_device = gallery.to(device=device, dtype=torch.float32)
    if reference_gallery is None and not model.normalize_components_before_fusion:
        raise ValueError('Raw reference embeddings are required by this head')
    reference_device = (gallery if reference_gallery is None else reference_gallery).to(device=device, dtype=torch.float32)
    candidate_indices = np.empty((len(records), top_k), dtype=np.int64)
    raw_target_in_topk = np.zeros(len(records), dtype=np.bool_)
    texts = [record.text for record in records]
    for begin in tqdm(
        range(0, len(records), config.mining.query_batch_size),
        desc=f"Student Top-{top_k} candidates epoch {epoch}",
    ):
        end = min(begin + config.mining.query_batch_size, len(records))
        text_inputs = processor(
            text=texts[begin:end],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with _autocast(device, config.optimizer.mixed_precision):
            text_features = model.encode_text(
                text_inputs["input_ids"].to(device),
                (
                    None
                    if text_inputs.get("attention_mask") is None
                    else text_inputs["attention_mask"].to(device)
                ),
            )
            queries = model.compose(
                reference_device[
                    torch.from_numpy(reference_indices[begin:end]).to(device)
                ],
                text_features,
            )
        scores = queries.float() @ gallery_device.t()
        local_references = torch.from_numpy(reference_indices[begin:end]).to(device)
        scores.scatter_(1, local_references[:, None], -torch.inf)
        raw_topk = torch.topk(scores, k=top_k, dim=1).indices.cpu().numpy()
        _force_targets_for_slice(
            candidate_indices=candidate_indices,
            raw_target_in_topk=raw_target_in_topk,
            raw_topk=raw_topk,
            target_indices=target_indices[begin:end],
            begin=begin,
            top_k=top_k,
            force_targets=config.mining.force_target_insertion,
        )
    del gallery_device
    return candidate_indices, raw_target_in_topk


@torch.inference_mode()
def _mine_teacher_topk(
    *,
    config: OnPolicyDistillationConfig,
    records: list[CIRRPositiveRecord],
    image_ids: list[str],
    reference_indices: np.ndarray,
    target_indices: np.ndarray,
    epoch: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    teacher_gallery, teacher_queries = _load_teacher_cache(
        config=config,
        records=records,
        image_ids=image_ids,
    )
    top_k = config.mining.top_k
    gallery_device = torch.from_numpy(
        np.asarray(teacher_gallery, dtype=np.float32).copy()
    ).to(device)
    candidate_indices = np.empty((len(records), top_k), dtype=np.int64)
    raw_target_in_topk = np.zeros(len(records), dtype=np.bool_)
    for begin in tqdm(
        range(0, len(records), config.mining.query_batch_size),
        desc=f"Teacher Top-{top_k} candidates epoch {epoch}",
    ):
        end = min(begin + config.mining.query_batch_size, len(records))
        queries = torch.from_numpy(
            np.asarray(teacher_queries[begin:end], dtype=np.float32).copy()
        ).to(device)
        scores = queries @ gallery_device.t()
        local_references = torch.from_numpy(reference_indices[begin:end]).to(device)
        scores.scatter_(1, local_references[:, None], -torch.inf)
        raw_topk = torch.topk(scores, k=top_k, dim=1).indices.cpu().numpy()
        _force_targets_for_slice(
            candidate_indices=candidate_indices,
            raw_target_in_topk=raw_target_in_topk,
            raw_topk=raw_topk,
            target_indices=target_indices[begin:end],
            begin=begin,
            top_k=top_k,
        )
    del gallery_device
    return candidate_indices, raw_target_in_topk


def _sample_random_candidates(
    *,
    sample_count: int,
    gallery_count: int,
    reference_indices: np.ndarray,
    target_indices: np.ndarray,
    top_k: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample K-1 unique non-targets without allocating a full pool per row."""

    if gallery_count - 2 < top_k - 1:
        raise ValueError("gallery is too small for random candidate sampling")
    rng = np.random.default_rng(seed)
    candidate_indices = np.empty((sample_count, top_k), dtype=np.int64)
    candidate_indices[:, 0] = target_indices
    for index, (reference_index, target_index) in enumerate(
        zip(reference_indices, target_indices, strict=True)
    ):
        compressed = rng.choice(gallery_count - 2, size=top_k - 1, replace=False)
        sampled = compressed.astype(np.int64, copy=True)
        for excluded in sorted((int(reference_index), int(target_index))):
            sampled += sampled >= excluded
        candidate_indices[index, 1:] = sampled
    raw_target_in_topk = np.zeros(sample_count, dtype=np.bool_)
    return candidate_indices, raw_target_in_topk


def _force_targets_for_slice(
    *,
    candidate_indices: np.ndarray,
    raw_target_in_topk: np.ndarray,
    raw_topk: np.ndarray,
    target_indices: np.ndarray,
    begin: int,
    top_k: int,
    force_targets: bool = True,
) -> None:
    for offset, (raw_candidates, target_index) in enumerate(
        zip(raw_topk, target_indices, strict=True)
    ):
        forced_candidates, contains_target = force_target_into_topk(
            raw_candidates,
            int(target_index),
            top_k,
        )
        raw_target_in_topk[begin + offset] = contains_target
        candidate_indices[begin + offset] = forced_candidates if force_targets else raw_candidates[:top_k]


def _load_teacher_cache(
    *,
    config: OnPolicyDistillationConfig,
    records: list[CIRRPositiveRecord],
    image_ids: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    cache_dir = config.data.teacher_cache_dir
    if cache_dir is None:
        raise ValueError("embedding Teacher cache directory is not configured")
    metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
    if not metadata.get("complete"):
        raise RuntimeError(f"Teacher on-policy cache is incomplete: {cache_dir}")
    cached_image_ids = json.loads(
        (cache_dir / "image_ids.json").read_text(encoding="utf-8")
    )
    cached_pair_ids = json.loads(
        (cache_dir / "pair_ids.json").read_text(encoding="utf-8")
    )
    if cached_image_ids != image_ids:
        raise ValueError("Teacher gallery order does not match Student gallery")
    if cached_pair_ids != [record.pair_id for record in records]:
        raise ValueError("Teacher query order does not match CIRR train annotations")
    teacher_gallery = np.load(cache_dir / "gallery_features.npy", mmap_mode="r")
    teacher_queries = np.load(cache_dir / "query_features.npy", mmap_mode="r")
    if teacher_gallery.ndim != 2 or teacher_queries.ndim != 2:
        raise ValueError("Teacher cache has invalid feature arrays")
    if teacher_gallery.shape[0] != len(image_ids) or teacher_queries.shape[0] != len(
        records
    ):
        raise ValueError("Teacher cache shape does not match CIRR train")
    return teacher_gallery, teacher_queries


def _load_teacher_negative_query_scores(
    config: OnPolicyDistillationConfig,
    negative_layout: NegativeQueryLayout,
) -> np.ndarray:
    cache_dir = config.data.teacher_cache_dir
    if cache_dir is None:
        raise ValueError("mixed q- candidates require a Teacher cache directory")
    ids_path = cache_dir / "negative_query_ids.json"
    owners_path = cache_dir / "negative_query_owners.npy"
    scores_path = cache_dir / "negative_query_pair_scores.npy"
    missing = [
        str(path)
        for path in (ids_path, owners_path, scores_path)
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError(
            "S2 q- candidate cache is incomplete; run "
            "scripts/prepare_on_policy_teacher_cache.py first. Missing: "
            + ", ".join(missing)
        )
    cached_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    cached_owners = np.load(owners_path, mmap_mode="r")
    cached_scores = np.load(scores_path, mmap_mode="r")
    if cached_ids != negative_layout.ids:
        raise ValueError("Teacher q- IDs do not match the configured manifest")
    if not np.array_equal(cached_owners, negative_layout.owners):
        raise ValueError("Teacher q- owners do not match the configured manifest")
    if cached_scores.shape != (len(negative_layout.ids),):
        raise ValueError("Teacher q- pair scores have an invalid shape")
    values = np.asarray(cached_scores, dtype=np.float32)
    if not bool(np.isfinite(values).all()):
        raise ValueError("Teacher q- pair scores contain non-finite values")
    return values


def _score_teacher_candidates(
    *,
    config: OnPolicyDistillationConfig,
    candidate_indices: np.ndarray,
    records: list[CIRRPositiveRecord],
    image_ids: list[str],
    image_paths: list[Path],
    output_dir: Path,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    if config.teacher.backend == "http_reranker":
        return _score_http_reranker_candidates(
            config=config,
            candidate_indices=candidate_indices,
            records=records,
            image_paths=image_paths,
            output_dir=output_dir,
        )
    teacher_gallery, teacher_queries = _load_teacher_cache(
        config=config,
        records=records,
        image_ids=image_ids,
    )
    result = np.empty(candidate_indices.shape, dtype=np.float32)
    top1 = np.zeros(len(records), dtype=np.bool_)
    for begin in tqdm(
        range(0, len(records), config.mining.teacher_score_batch_size),
        desc="Teacher scores on Student candidates",
    ):
        end = min(begin + config.mining.teacher_score_batch_size, len(records))
        queries = torch.from_numpy(
            np.asarray(teacher_queries[begin:end], dtype=np.float32).copy()
        ).to(device)
        candidates = torch.from_numpy(
            np.asarray(
                teacher_gallery[candidate_indices[begin:end]], dtype=np.float32
            ).copy()
        ).to(device)
        with _autocast(device, config.optimizer.mixed_precision):
            scores = torch.einsum("bd,bkd->bk", queries, candidates)
        scores_cpu = scores.float().cpu().numpy()
        result[begin:end] = scores_cpu
        top1[begin:end] = scores_cpu.argmax(axis=1) == 0
    return result, top1


def _score_http_reranker_candidates(
    *,
    config: OnPolicyDistillationConfig,
    candidate_indices: np.ndarray,
    records: list[CIRRPositiveRecord],
    image_paths: list[Path],
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Score one Student candidate set per request with resumable row caching."""

    partial_path = output_dir / "teacher_scores_http.partial.npy"
    progress_path = output_dir / "teacher_scores_http.progress.json"
    fingerprint = _http_scoring_fingerprint(config, candidate_indices, records)
    next_index = 0
    score_cache: np.memmap | None = None
    if partial_path.is_file() or progress_path.is_file():
        if partial_path.is_file() and progress_path.is_file():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            loaded = np.load(partial_path, mmap_mode="r+")
            if (
                progress.get("fingerprint") == fingerprint
                and loaded.shape == candidate_indices.shape
                and loaded.dtype == np.float32
            ):
                score_cache = loaded
                next_index = int(progress.get("next_index", 0))
                if next_index < 0 or next_index > len(records):
                    raise ValueError("invalid HTTP reranker progress index")
                print(
                    f"[resume reranker scores] {next_index}/{len(records)} rows",
                    flush=True,
                )
        if score_cache is None:
            partial_path.unlink(missing_ok=True)
            progress_path.unlink(missing_ok=True)
    if score_cache is None:
        score_cache = np.lib.format.open_memmap(
            partial_path,
            mode="w+",
            dtype=np.float32,
            shape=candidate_indices.shape,
        )
        write_json_atomic(
            progress_path,
            {
                "complete": False,
                "fingerprint": fingerprint,
                "next_index": 0,
                "sample_count": len(records),
                "top_k": int(candidate_indices.shape[1]),
            },
        )

    client = HttpRerankerClient(config.teacher)

    def score_row(index: int) -> np.ndarray:
        paths = [image_paths[int(value)] for value in candidate_indices[index]]
        return client.score_candidates(record=records[index], candidate_paths=paths)

    concurrency = config.teacher.max_concurrent_requests
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        for begin in tqdm(
            range(next_index, len(records), concurrency),
            desc="HTTP reranker Teacher scores",
        ):
            end = min(begin + concurrency, len(records))
            rows = list(executor.map(score_row, range(begin, end)))
            values = np.stack(rows).astype(np.float32, copy=False)
            if values.shape != (end - begin, candidate_indices.shape[1]):
                raise ValueError("HTTP reranker returned an invalid score matrix")
            score_cache[begin:end] = values
            score_cache.flush()
            write_json_atomic(
                progress_path,
                {
                    "complete": end == len(records),
                    "fingerprint": fingerprint,
                    "next_index": end,
                    "sample_count": len(records),
                    "top_k": int(candidate_indices.shape[1]),
                },
            )
    result = np.asarray(score_cache, dtype=np.float32).copy()
    if not bool(np.isfinite(result).all()):
        raise ValueError("HTTP reranker score cache contains non-finite values")
    top1 = result.argmax(axis=1) == 0
    return result, top1


def _http_scoring_fingerprint(
    config: OnPolicyDistillationConfig,
    candidate_indices: np.ndarray,
    records: list[CIRRPositiveRecord],
) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(candidate_indices).tobytes())
    digest.update(
        json.dumps(
            [record.pair_id for record in records], separators=(",", ":")
        ).encode("utf-8")
    )
    digest.update(
        json.dumps(
            config.teacher.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _teacher_identity(config: OnPolicyDistillationConfig) -> dict[str, Any]:
    if config.teacher.backend == "http_reranker":
        return config.teacher.model_dump(mode="json")
    return {
        "backend": "embedding_cache",
        "teacher_cache_dir": (
            None
            if config.data.teacher_cache_dir is None
            else str(config.data.teacher_cache_dir)
        ),
    }


def _teacher_score_statistics(
    scores: np.ndarray,
    *,
    temperature: float,
) -> dict[str, float]:
    values = np.asarray(scores, dtype=np.float64)
    logits = values / temperature
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    entropy = -(probabilities * np.log(np.maximum(probabilities, 1e-300))).sum(
        axis=1
    )
    unique_per_row = np.asarray(
        [len(np.unique(row)) for row in values], dtype=np.float64
    )
    return {
        "teacher_score_min": float(values.min()),
        "teacher_score_max": float(values.max()),
        "teacher_score_mean": float(values.mean()),
        "teacher_score_std": float(values.std()),
        "teacher_score_exact_zero_rate": float((values == 0.0).mean()),
        "teacher_score_exact_one_rate": float((values == 1.0).mean()),
        "teacher_score_unique_per_row_mean": float(unique_per_row.mean()),
        "teacher_rank_temperature": float(temperature),
        "teacher_rank_entropy": float(entropy.mean()),
    }


def _save_array_atomic(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    temporary.replace(path)


def force_target_into_topk(
    raw_candidates: np.ndarray,
    target_index: int,
    top_k: int,
) -> tuple[np.ndarray, bool]:
    """Return exactly K unique candidates with target fixed at position zero."""

    values = np.asarray(raw_candidates, dtype=np.int64)
    if values.ndim != 1 or len(values) < top_k:
        raise ValueError("raw Top-K candidates have an invalid shape")
    if len(set(values[:top_k].tolist())) != top_k:
        raise ValueError("raw Top-K candidates contain duplicates")
    contains_target = bool((values[:top_k] == target_index).any())
    non_targets = values[values != target_index]
    if len(non_targets) < top_k - 1:
        raise ValueError("raw Top-K candidates have too few non-target images")
    result = np.empty(top_k, dtype=np.int64)
    result[0] = target_index
    result[1:] = non_targets[: top_k - 1]
    return result, contains_target


def _load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _chunks(values: list[Path], size: int) -> Iterable[list[Path]]:
    for begin in range(0, len(values), size):
        yield values[begin : begin + size]


def _autocast(device: torch.device, mixed_precision: str):
    from contextlib import nullcontext

    if device.type != "cuda" or mixed_precision == "no":
        return nullcontext()
    dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)
