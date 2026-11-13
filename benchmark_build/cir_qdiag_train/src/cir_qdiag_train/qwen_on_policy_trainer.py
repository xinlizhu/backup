from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from cir_qdiag_train.on_policy_data import (
    CIRRPositiveRecord,
    load_cirr_positive_records,
    load_image_map,
)
from cir_qdiag_train.on_policy_loss import (
    on_policy_centered_distillation_loss,
    candidate_infonce_loss,
    centered_similarity_mse_loss,
)
from cir_qdiag_train.on_policy_mining import (
    _force_targets_for_slice,
    _save_array_atomic,
    _score_teacher_candidates,
    _sample_random_candidates,
    write_json_atomic,
)
from cir_qdiag_train.qwen_on_policy_config import QwenOnPolicyDistillationConfig
from cir_qdiag_train.qwen_training_data import query_batches, image_batches, shared_training_batches
from cir_qdiag_train.qwen_gradient_cache import shared_student_scores
from cir_qdiag_train.qwen_live_training import compute_live_qwen_scores
from cir_qdiag_train.qwen_student import (
    QwenQueryLoRAStudent,
    load_qwen_checkpoint,
    save_qwen_checkpoint,
)


def train_qwen_on_policy_distillation(
    config: QwenOnPolicyDistillationConfig,
    *,
    resume_from: str | Path | None = None,
) -> Path:
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or config.optimizer.world_size != 1:
        raise ValueError("Qwen OPD currently supports one process/GPU")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _set_seed(config.experiment.seed)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True

    output_dir = config.experiment.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "effective_config.json"
    if config_path.exists():
        from cir_qdiag_train.resume_config import without_loader_settings
        old = QwenOnPolicyDistillationConfig.model_validate(json.loads(config_path.read_text()))
        if without_loader_settings(old.model_dump(mode="json")) != without_loader_settings(config.model_dump(mode="json")):
            raise ValueError("Refusing changed experiment configuration in existing output directory")
        if resume_from is None and (output_dir / "last.pt").exists():
            raise ValueError("Existing checkpoint: use --resume-from or a fresh output directory")
    write_json_atomic(config_path, config.model_dump(mode="json"))
    image_ids, image_paths = load_image_map(
        config.data.train_image_split_path, config.data.image_root
    )
    records = load_cirr_positive_records(
        config.data.train_annotation_path, image_ids, image_paths
    )
    gallery = None if config.model.train_gallery else _load_student_gallery(config, image_ids)
    student_class = QwenQueryLoRAStudent
    if config.model.llava:
        from cir_qdiag_train.llava_student import LlavaLoRAStudent
        student_class = LlavaLoRAStudent
    elif config.model.qwen35:
        from cir_qdiag_train.qwen35_student import Qwen35Student
        student_class = Qwen35Student
    student = student_class(
        config.model,
        device=device,
        mixed_precision=config.optimizer.mixed_precision,
    )
    trainable = student.trainable_parameters()
    if not trainable:
        raise RuntimeError("Qwen Student has no trainable parameters")
    if config.model.train_gallery:
        audit = student.trainability_audit() if (config.model.qwen35 or config.model.llava) else audit_shared_adapters(student)
        write_json_atomic(output_dir / "adapter_audit.json", audit)
    optimizer = AdamW(
        trainable,
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
    )
    micro_steps_per_epoch = math.ceil(
        len(records) / config.optimizer.batch_size_per_gpu
    )
    steps_per_epoch = math.ceil(
        micro_steps_per_epoch / config.optimizer.gradient_accumulation_steps
    )
    total_steps = steps_per_epoch * config.optimizer.epochs
    scheduler = LambdaLR(
        optimizer,
        _cosine_schedule(
            total_steps=total_steps,
            warmup_steps=int(total_steps * config.optimizer.warmup_ratio),
        ),
    )
    start_epoch = 0
    global_step = 0
    use_fp16 = config.optimizer.mixed_precision == "fp16" and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    if resume_from is not None:
        checkpoint = load_qwen_checkpoint(
            resume_from,
            student=student,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected_config=config.model_dump(mode="json"),
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])

    write_json_atomic(
        output_dir / "run_summary.json",
        {
            "algorithm": config.objective.loss_type,
            "rank_kl_direction": config.objective.rank_kl_direction,
            "student_initialization": config.model.pretrained_model_name_or_path,
            "student_architecture": config.model.variant,
            "teacher_cache": str(config.data.teacher_cache_dir) if config.objective.needs_teacher_scores else None,
            "candidate_count_per_query": config.mining.top_k,
            "candidate_source": config.mining.candidate_source,
            "uses_teacher_scores": config.objective.needs_teacher_scores,
            "candidate_refresh": "every epoch",
            "target_forced_into_candidate_set": True,
            "teacher_target_max_rank": config.objective.teacher_target_max_rank,
            "image_encoder": ("full retrieval backbone" if config.model.variant == "qwen3_5_full" else
                              "shared all-linear LoRA" if config.model.train_gallery else "frozen original Qwen-2B; LoRA disabled"),
            "query_encoder": ("full retrieval backbone" if config.model.variant == "qwen3_5_full" else
                              "shared all-linear LoRA" if config.model.train_gallery else "original Qwen-2B plus language-attention LoRA"),
            "gallery_policy": "re-encoded each mining epoch; live differentiable candidates in loss" if config.model.train_gallery else "immutable exact deployment gallery",
            "candidate_gradient_cache": config.optimizer.candidate_gradient_cache if config.model.train_gallery else False,
            "backward_mode": ("embedding_gradient_replay" if config.optimizer.candidate_gradient_cache
                              else "direct_live_backward") if config.model.train_gallery else "query_only",
            "effective_batch_size": (
                config.optimizer.batch_size_per_gpu
                * config.optimizer.gradient_accumulation_steps
            ),
            "optimizer_steps_per_epoch": steps_per_epoch,
            "data_loader_workers": config.data.num_workers,
            "data_loader_prefetch_factor": config.data.prefetch_factor,
            "total_optimizer_steps": total_steps,
            **student.parameter_summary(),
        },
    )

    training_started = time.perf_counter()
    epoch_timings: list[dict[str, float | int]] = []
    gradient_audited = (output_dir / "gradient_audit.json").is_file()
    for epoch_index in range(start_epoch, config.optimizer.epochs):
        epoch_number = epoch_index + 1
        epoch_started = time.perf_counter()
        mining_started = time.perf_counter()
        mining_dir, mining_metadata = refresh_qwen_candidates(
            config=config,
            student=student,
            records=records,
            image_ids=image_ids,
            gallery=gallery,
            image_paths=image_paths,
            epoch=epoch_number,
            global_step=global_step,
            device=device,
        )
        mining_seconds = time.perf_counter() - mining_started
        if config.model.train_gallery:
            # Isolate training RNG from mining/cache reuse and worker startup.
            _set_seed(config.experiment.seed + epoch_index)
        candidates = np.load(mining_dir / "candidate_indices.npy", mmap_mode="r")
        raw_target = np.load(mining_dir / "raw_target_in_topk.npy", mmap_mode="r")
        teacher_scores = np.load(mining_dir / "teacher_scores.npy", mmap_mode="r")
        teacher_top1 = np.load(mining_dir / "teacher_target_top1.npy", mmap_mode="r")

        student.train()
        optimizer.zero_grad(set_to_none=True)
        order = torch.randperm(
            len(records),
            generator=torch.Generator().manual_seed(
                config.experiment.seed + epoch_index
            ),
        ).tolist()
        running: dict[str, float] = {}
        running_count = 0
        training_phase_started = time.perf_counter()
        batch_size = config.optimizer.batch_size_per_gpu
        batch_kwargs = dict(
            student=student, records=records, order=order, batch_size=batch_size,
            workers=config.data.num_workers, prefetch_factor=config.data.prefetch_factor,
            seed=config.experiment.seed + epoch_index,
        )
        if config.model.train_gallery:
            batches = shared_training_batches(
                **batch_kwargs, candidates=candidates, image_paths=image_paths,
                image_batch_size=config.optimizer.candidate_encode_batch_size,
            )
            batches = tqdm(batches, total=micro_steps_per_epoch,
                           desc=f"Qwen {config.model.variant} train epoch {epoch_number}")
        else:
            batches = query_batches(**batch_kwargs)
        for micro_step, (indices, prepared) in enumerate(batches):
            begin = micro_step * batch_size
            replay = None
            if config.model.train_gallery:
                if config.optimizer.candidate_gradient_cache:
                    student_scores, replay = shared_student_scores(student, prepared, gradient_cache=True)
                else:
                    # Same training pattern as CLIP's _compute_live_student_scores:
                    # live candidate features -> loss -> one ordinary backward.
                    student_scores = compute_live_qwen_scores(student, prepared)
            else:
                queries = student.encode_prepared_queries(prepared, requires_grad=True)
                candidate_features = torch.from_numpy(
                    np.asarray(gallery[np.asarray(candidates[indices])], dtype=np.float32).copy()
                ).to(device)
                student_scores = torch.einsum("bd,bkd->bk", queries, candidate_features)
            loss, metrics = qwen_candidate_loss(
                config=config,
                student_scores=student_scores,
                teacher_scores=torch.from_numpy(
                    np.asarray(teacher_scores[indices], dtype=np.float32).copy()
                ).to(device),
                teacher_target_top1=torch.from_numpy(
                    np.asarray(teacher_top1[indices], dtype=np.bool_).copy()
                ).to(device),
                raw_target_in_topk=torch.from_numpy(
                    np.asarray(raw_target[indices], dtype=np.bool_).copy()
                ).to(device),
                rank_weight=config.objective.rank_weight,
                centered_response_weight=0.0,
                rank_temperature=config.objective.rank_temperature,
                centered_response_beta=config.objective.centered_response_beta,
                require_teacher_target_top1=(
                    config.objective.require_teacher_target_top1
                ),
                teacher_target_max_rank=config.objective.teacher_target_max_rank,
                rank_kl_direction=config.objective.rank_kl_direction,
            )
            # The final accumulation group can contain fewer microbatches.
            group_start = (micro_step // config.optimizer.gradient_accumulation_steps) * config.optimizer.gradient_accumulation_steps
            group_size = min(config.optimizer.gradient_accumulation_steps, micro_steps_per_epoch - group_start)
            scaled_loss = loss / (group_size if config.model.train_gallery else config.optimizer.gradient_accumulation_steps)
            scaler.scale(scaled_loss).backward()
            if replay is not None:
                replay()
            if not gradient_audited:
                write_json_atomic(
                    output_dir / "gradient_audit.json",
                    _audit_gradients(student),
                )
                gradient_audited = True
            for key, value in metrics.items():
                running[key] = running.get(key, 0.0) + float(value.item())
            running_count += 1
            sync_step = (
                (micro_step + 1) % config.optimizer.gradient_accumulation_steps == 0
                or begin + batch_size >= len(order)
            )
            if not sync_step:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, config.optimizer.max_grad_norm,
                                           error_if_nonfinite=config.model.qwen35 or config.model.llava)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            if global_step % config.optimizer.log_every_steps == 0:
                payload: dict[str, Any] = {
                    "epoch_number": epoch_number,
                    "global_step": global_step,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "mining_raw_target_recall_at_k": mining_metadata[
                        "raw_target_recall_at_k"
                    ],
                    "mining_teacher_target_top10_rate": mining_metadata[
                        "teacher_target_top10_rate"
                    ],
                    **{
                        key: round(value / running_count, 7)
                        for key, value in running.items()
                    },
                }
                line = json.dumps(payload, ensure_ascii=False)
                print(line, flush=True)
                with (output_dir / "train_metrics.jsonl").open(
                    "a", encoding="utf-8"
                ) as handle:
                    handle.write(line + "\n")
                running.clear()
                running_count = 0

        training_seconds = time.perf_counter() - training_phase_started
        epoch_seconds = time.perf_counter() - epoch_started
        epoch_timings.append(
            {
                "epoch": epoch_number,
                "mining_seconds": mining_seconds,
                "training_seconds": training_seconds,
                "total_seconds": epoch_seconds,
            }
        )
        write_json_atomic(
            output_dir / "training_time.json",
            {
                "completed_epochs": epoch_number,
                "elapsed_seconds_this_process": time.perf_counter()
                - training_started,
                "epochs_this_process": epoch_timings,
            },
        )
        if (
            epoch_number % config.optimizer.save_every_epochs == 0
            or epoch_number == config.optimizer.epochs
        ):
            config_payload = config.model_dump(mode="json")
            save_qwen_checkpoint(
                output_dir / f"checkpoint_epoch_{epoch_number:03d}.pt",
                student=student,
                optimizer=None,
                scheduler=None,
                epoch=epoch_index,
                global_step=global_step,
                config=config_payload,
                scaler=scaler,
            )
            save_qwen_checkpoint(
                output_dir / "last.pt",
                student=student,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch_index,
                global_step=global_step,
                config=config_payload,
                scaler=scaler,
            )
    return output_dir / "last.pt"


def qwen_candidate_loss(*, config, **kwargs):
    """Keep the legacy Rank path intact; baseline losses use the same score support."""
    common = {key: kwargs[key] for key in (
        "student_scores", "raw_target_in_topk", "require_teacher_target_top1",
        "teacher_target_max_rank",
    )}
    if config.objective.loss_type == "candidate_infonce":
        return candidate_infonce_loss(
            **common, temperature=config.objective.rank_temperature,
            teacher_scores=kwargs["teacher_scores"] if config.objective.uses_teacher_filter else None,
        )
    if config.objective.loss_type == "centered_mse":
        return centered_similarity_mse_loss(**common, teacher_scores=kwargs["teacher_scores"])
    return on_policy_centered_distillation_loss(**kwargs)


@torch.inference_mode()
def refresh_qwen_candidates(
    *,
    config: QwenOnPolicyDistillationConfig,
    student: QwenQueryLoRAStudent,
    records: list[CIRRPositiveRecord],
    image_ids: list[str],
    gallery: np.ndarray | None,
    epoch: int,
    global_step: int,
    device: torch.device,
    image_paths: list[Path] | None = None,
) -> tuple[Path, dict[str, Any]]:
    output_dir = config.experiment.output_dir / "on_policy" / f"epoch_{epoch:03d}"
    metadata_path = output_dir / "metadata.json"
    required = [
        output_dir / "candidate_indices.npy",
        output_dir / "target_indices.npy",
        output_dir / "raw_target_in_topk.npy",
        output_dir / "teacher_scores.npy",
        output_dir / "teacher_target_top1.npy",
    ]
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("complete")
            and int(metadata.get("source_global_step", -1)) == global_step
            and int(metadata.get("top_k", -1)) == config.mining.top_k
            and metadata.get("candidate_source", "student") == config.mining.candidate_source
            and metadata.get("uses_teacher_scores", True) == config.objective.needs_teacher_scores
            and metadata.get("shared_lora", False) == config.model.shared_lora
            and (not config.model.qwen35 or metadata.get("student_variant") == config.model.variant)
            and all(path.is_file() for path in required)
        ):
            print(f"[reuse mining] {output_dir}", flush=True)
            return output_dir, metadata

    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    student.eval()
    image_index = {image_id: index for index, image_id in enumerate(image_ids)}
    reference_indices = np.asarray(
        [image_index[record.reference_id] for record in records], dtype=np.int64
    )
    target_indices = np.asarray(
        [image_index[record.target_id] for record in records], dtype=np.int64
    )
    is_random = config.mining.candidate_source == "random"
    if config.model.train_gallery and not is_random:
        if image_paths is None or len(image_paths) != len(image_ids):
            raise ValueError("shared LoRA mining requires current gallery image paths")
        gallery = encode_current_gallery(config, student, image_paths, epoch)
    if not is_random and gallery is None:
        raise ValueError("Student mining needs gallery features")
    gallery_device = None if is_random else torch.from_numpy(
        np.asarray(gallery, dtype=np.float32).copy()
    ).to(device)
    top_k = config.mining.top_k
    candidate_indices = np.empty((len(records), top_k), dtype=np.int64)
    raw_target_in_topk = np.zeros(len(records), dtype=np.bool_)
    if is_random:
        candidate_indices, raw_target_in_topk = _sample_random_candidates(
            sample_count=len(records), gallery_count=len(image_ids),
            reference_indices=reference_indices, target_indices=target_indices,
            top_k=top_k, seed=config.experiment.seed + epoch,
        )
    query_batch_size = config.mining.query_batch_size
    batches = () if is_random else query_batches(
        student=student, records=records, order=list(range(len(records))),
        batch_size=query_batch_size, workers=config.data.num_workers,
        prefetch_factor=config.data.prefetch_factor, seed=config.experiment.seed + epoch,
    )
    for indices, prepared in tqdm(
        batches, total=0 if is_random else math.ceil(len(records) / query_batch_size),
        desc=f"Qwen Student Top-{top_k} candidates epoch {epoch}",
    ):
        begin, end = indices[0], indices[-1] + 1
        queries = student.encode_prepared_queries(prepared, requires_grad=False)
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
        )
    del gallery_device
    if not np.array_equal(candidate_indices[:, 0], target_indices):
        raise RuntimeError("forced Qwen candidates lost a target")
    _save_array_atomic(output_dir / "candidate_indices.npy", candidate_indices)
    _save_array_atomic(output_dir / "target_indices.npy", target_indices)
    _save_array_atomic(output_dir / "raw_target_in_topk.npy", raw_target_in_topk)
    teacher_scores, teacher_target_top1 = _score_teacher_candidates(
        config=config,
        candidate_indices=candidate_indices,
        records=records,
        image_ids=image_ids,
        image_paths=[],  # embedding-cache backend does not open gallery images
        output_dir=output_dir,
        device=device,
    ) if config.objective.needs_teacher_scores else (
        np.zeros(candidate_indices.shape, dtype=np.float32),
        np.zeros(len(records), dtype=np.bool_),
    )
    cache_dtype = np.float16 if config.mining.cache_dtype == "float16" else np.float32
    _save_array_atomic(
        output_dir / "teacher_scores.npy",
        teacher_scores.astype(cache_dtype, copy=False),
    )
    _save_array_atomic(
        output_dir / "teacher_target_top1.npy", teacher_target_top1
    )
    teacher_ranks = 1 + (
        teacher_scores[:, 1:] > teacher_scores[:, :1]
    ).sum(axis=1)
    metadata = {
        "schema_version": 1,
        "complete": True,
        "epoch": epoch,
        "source_global_step": global_step,
        "sample_count": len(records),
        "gallery_count": len(image_ids),
        "top_k": top_k,
        "candidate_source": config.mining.candidate_source,
        "uses_teacher_scores": config.objective.needs_teacher_scores,
        "student_policy": ("full backbone; epoch-current gallery" if config.model.variant == "qwen3_5_full" else
                           "shared all-linear LoRA; epoch-current gallery" if config.model.train_gallery else
                           "Qwen query LoRA against immutable base gallery"),
        "shared_lora": config.model.shared_lora,
        "student_variant": config.model.variant,
        "target_forced_into_candidates": True,
        "raw_target_recall_at_k": float(raw_target_in_topk.mean()),
        "teacher_target_top1_rate": float(teacher_target_top1.mean()) if config.objective.needs_teacher_scores else None,
        "teacher_target_top10_rate": float((teacher_ranks <= 10).mean()) if config.objective.needs_teacher_scores else None,
        "teacher_target_mean_rank": float(teacher_ranks.mean()) if config.objective.needs_teacher_scores else None,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json_atomic(metadata_path, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output_dir, metadata


@torch.inference_mode()
def encode_current_gallery(config, student, image_paths, epoch):
    student.eval()
    features = []
    batches = image_batches(
        student=student, image_paths=image_paths, batch_size=config.mining.image_batch_size,
        workers=config.data.num_workers, prefetch_factor=config.data.prefetch_factor,
        seed=config.experiment.seed + epoch,
    )
    for _, prepared in tqdm(batches, total=math.ceil(len(image_paths) / config.mining.image_batch_size),
                            desc=f"Shared-LoRA Student gallery epoch {epoch}"):
        features.append(student.encode_prepared_images(prepared).float().cpu().numpy())
    return np.concatenate(features)


def audit_shared_adapters(student):
    names = [name for name, module in student.model.named_modules() if hasattr(module, "lora_A")]
    unadapted = [
        name for name, module in student.model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and not any(name.startswith(parent + ".") for parent in names)
    ]
    if unadapted:
        raise RuntimeError(f"Linear modules missing LoRA adapters: {unadapted}")
    groups = {
        "language": [n for n in names if "language_model" in n],
        "vision": [n for n in names if "visual.blocks" in n],
        "merger": [n for n in names if "merger" in n],
    }
    if any(not group for group in groups.values()):
        raise RuntimeError(f"all-linear failed to cover every Qwen component: {groups}")
    unexpected = [n for n, p in student.model.named_parameters() if p.requires_grad and "lora_" not in n]
    if unexpected:
        raise RuntimeError(f"Non-LoRA parameters unexpectedly trainable: {unexpected}")
    return {"target_modules": "all-linear", "shared_query_gallery": True,
            "modules": groups, **student.parameter_summary()}


def _load_student_gallery(
    config: QwenOnPolicyDistillationConfig, image_ids: list[str]
) -> np.ndarray:
    cache_dir = config.data.student_gallery_cache_dir
    metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
    if not metadata.get("complete"):
        raise RuntimeError(f"incomplete Qwen Student gallery: {cache_dir}")
    if config.model.variant == "qwen3_vl_instruct_query_lora":
        expected_preprocessing = {
            "max_length": config.model.max_length,
            "min_pixels": config.model.min_pixels,
            "max_pixels": config.model.max_pixels,
            "instruction": QwenQueryLoRAStudent.TARGET_INSTRUCTION,
        }
        if (metadata.get("student_type") != config.model.variant
                or metadata.get("base_model_path") != str(Path(config.model.pretrained_model_name_or_path).resolve())
                or metadata.get("preprocessing") != expected_preprocessing):
            raise ValueError("Instruct gallery does not match the base model/preprocessing; do not reuse Embedding features")
    cached_ids = json.loads((cache_dir / "image_ids.json").read_text(encoding="utf-8"))
    if cached_ids != image_ids:
        raise ValueError("Qwen Student gallery image order does not match CIRR train")
    gallery = np.load(cache_dir / "gallery_features.npy", mmap_mode="r")
    if gallery.ndim != 2 or gallery.shape[0] != len(image_ids):
        raise ValueError("Qwen Student gallery cache has an invalid shape")
    return gallery


def _audit_gradients(student: QwenQueryLoRAStudent) -> dict[str, int]:
    parameters = student.trainable_parameters()
    return {
        "trainable_tensor_count": len(parameters),
        "trainable_parameter_count": sum(p.numel() for p in parameters),
        "nonzero_gradient_tensor_count": sum(
            p.grad is not None and bool((p.grad != 0).any()) for p in parameters
        ),
        "missing_gradient_count": sum(p.grad is None for p in parameters),
        "nonfinite_gradient_count": sum(
            p.grad is not None and not bool(torch.isfinite(p.grad).all())
            for p in parameters
        ),
    }


def _cosine_schedule(*, total_steps: int, warmup_steps: int):
    def schedule(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return schedule


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
