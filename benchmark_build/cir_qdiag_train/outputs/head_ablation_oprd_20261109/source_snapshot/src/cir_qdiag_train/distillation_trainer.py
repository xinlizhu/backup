from __future__ import annotations

import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from cir_qdiag_train.distillation_config import SemanticDistillationConfig
from cir_qdiag_train.distillation_data import (
    SemanticDistillationCollator,
    SemanticDistillationDataset,
)
from cir_qdiag_train.distillation_loss import semantic_ability_distillation_loss
from cir_qdiag_train.model import ComposedRetrievalModel, load_checkpoint, save_checkpoint


def train_semantic_distillation(
    config: SemanticDistillationConfig,
    *,
    resume_from: str | Path | None = None,
) -> Path:
    distributed, rank, local_rank, world_size = _setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    _set_seed(config.experiment.seed + rank)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True

    output_dir = config.experiment.output_dir
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "effective_config.json").write_text(
            json.dumps(config.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

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
    if resume_from is None:
        load_checkpoint(config.experiment.initialize_from, model=model)
    model.to(device)

    dataset = SemanticDistillationDataset(config)
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=config.experiment.seed,
            drop_last=False,
        )
        if distributed
        else None
    )
    generator = torch.Generator().manual_seed(config.experiment.seed + rank)
    loader = DataLoader(
        dataset,
        batch_size=config.optimizer.batch_size_per_gpu,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=config.data.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=config.data.num_workers > 0,
        collate_fn=SemanticDistillationCollator(
            processor,
            include_negative_queries=config.objective.semantic_response_weight > 0,
        ),
        worker_init_fn=_seed_worker,
        generator=generator,
    )
    if len(loader) == 0:
        raise ValueError("semantic distillation loader is empty")

    optimizer = _create_optimizer(model, config)
    update_steps_per_epoch = math.ceil(
        len(loader) / config.optimizer.gradient_accumulation_steps
    )
    total_steps = update_steps_per_epoch * config.optimizer.epochs
    scheduler = LambdaLR(
        optimizer,
        _cosine_schedule(
            total_steps=total_steps,
            warmup_steps=int(total_steps * config.optimizer.warmup_ratio),
        ),
    )
    start_epoch = 0
    global_step = 0
    if resume_from is not None:
        checkpoint = load_checkpoint(
            resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])

    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    if rank == 0:
        summary = {
            "algorithm": "Teacher positive-rank / semantic-response distillation",
            "uses_infonce": False,
            "num_train_samples": len(dataset),
            "manifest_sample_count": len(dataset.records),
            "positive_rank_weight": config.objective.positive_rank_weight,
            "semantic_response_weight": config.objective.semantic_response_weight,
            "semantic_target": config.data.semantic_target,
            "edit_operations": config.data.edit_operations,
            "student_initialize_from": str(config.experiment.initialize_from),
            "student_image_encoder": "frozen and precomputed",
            "world_size": world_size,
            "micro_steps_per_epoch_per_rank": len(loader),
            "optimizer_steps_per_epoch": update_steps_per_epoch,
            "total_optimizer_steps": total_steps,
            **_unwrap(model).parameter_summary(),
        }
        (output_dir / "run_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    use_fp16 = config.optimizer.mixed_precision == "fp16" and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    autocast_dtype = (
        torch.bfloat16
        if config.optimizer.mixed_precision == "bf16"
        else torch.float16
    )
    optimizer.zero_grad(set_to_none=True)
    gradient_audited = False
    training_started = time.perf_counter()
    epoch_times: list[float] = []
    for epoch in range(start_epoch, config.optimizer.epochs):
        epoch_started = time.perf_counter()
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        running: dict[str, float] = {}
        running_count = 0
        for micro_step, batch in enumerate(loader):
            batch = _move_batch(batch, device)
            sync_step = (
                (micro_step + 1) % config.optimizer.gradient_accumulation_steps == 0
                or micro_step + 1 == len(loader)
            )
            sync_context = (
                nullcontext()
                if sync_step or not isinstance(model, DistributedDataParallel)
                else model.no_sync()
            )
            amp_context = (
                torch.autocast(device_type="cuda", dtype=autocast_dtype)
                if device.type == "cuda" and config.optimizer.mixed_precision != "no"
                else nullcontext()
            )
            with sync_context, amp_context:
                raw_model = _unwrap(model)
                positive_text_features = raw_model.encode_text(
                    batch["positive_input_ids"],
                    batch["positive_attention_mask"],
                )
                positive_queries = raw_model.compose(
                    F.normalize(batch["reference_features"], dim=-1),
                    positive_text_features,
                )
                negative_queries = None
                if batch["negative_input_ids"] is not None:
                    negative_text_features = raw_model.encode_text(
                        batch["negative_input_ids"],
                        batch["negative_attention_mask"],
                    )
                    negative_queries = raw_model.compose(
                        F.normalize(
                            batch["reference_features"][batch["negative_owner"]],
                            dim=-1,
                        ),
                        negative_text_features,
                    )
                loss, metrics = semantic_ability_distillation_loss(
                    positive_queries=positive_queries,
                    negative_queries=negative_queries,
                    negative_owner=batch["negative_owner"],
                    candidate_features=F.normalize(batch["candidate_features"], dim=-1),
                    candidate_mask=batch["candidate_mask"],
                    target_indices=batch["target_indices"],
                    teacher_positive_scores=batch["teacher_positive_scores"],
                    teacher_negative_scores=batch["teacher_negative_scores"],
                    teacher_target_top1=batch["teacher_target_top1"],
                    positive_rank_weight=config.objective.positive_rank_weight,
                    semantic_response_weight=config.objective.semantic_response_weight,
                    positive_rank_temperature=(
                        config.objective.positive_rank_temperature
                    ),
                    semantic_response_beta=config.objective.semantic_response_beta,
                    require_teacher_target_top1=(
                        config.objective.require_teacher_target_top1
                    ),
                    semantic_min_teacher_target_delta=(
                        config.objective.semantic_min_teacher_target_delta
                    ),
                )
                scaled_loss = loss / config.optimizer.gradient_accumulation_steps
                scaler.scale(scaled_loss).backward()

            if not gradient_audited:
                audit = _audit_trainable_gradients(_unwrap(model))
                if rank == 0:
                    (output_dir / "gradient_audit.json").write_text(
                        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                gradient_audited = True
            for key, value in metrics.items():
                running[key] = running.get(key, 0.0) + float(value.item())
            running_count += 1
            if not sync_step:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                _unwrap(model).parameters(), config.optimizer.max_grad_norm
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            if rank == 0 and global_step % config.optimizer.log_every_steps == 0:
                learning_rates = scheduler.get_last_lr()
                payload: dict[str, Any] = {
                    "epoch_number": epoch + 1,
                    "global_step": global_step,
                    "head_learning_rate": learning_rates[0],
                    "text_backbone_learning_rate": (
                        learning_rates[1] if len(learning_rates) > 1 else None
                    ),
                }
                payload.update(
                    {
                        key: round(value / running_count, 7)
                        for key, value in running.items()
                    }
                )
                text = json.dumps(payload, ensure_ascii=False)
                print(text, flush=True)
                with (output_dir / "train_metrics.jsonl").open(
                    "a", encoding="utf-8"
                ) as handle:
                    handle.write(text + "\n")
                running.clear()
                running_count = 0

        epoch_times.append(time.perf_counter() - epoch_started)
        if distributed:
            dist.barrier()
        if rank == 0 and (
            (epoch + 1) % config.optimizer.save_every_epochs == 0
            or epoch + 1 == config.optimizer.epochs
        ):
            raw_model = _unwrap(model)
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1:03d}.pt"
            save_checkpoint(
                checkpoint_path,
                model=raw_model,
                optimizer=None,
                scheduler=None,
                epoch=epoch,
                global_step=global_step,
                config=config.model_dump(mode="json"),
                base_checkpoint=config.experiment.initialize_from,
            )
            save_checkpoint(
                output_dir / "last.pt",
                model=raw_model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                config=config.model_dump(mode="json"),
                base_checkpoint=config.experiment.initialize_from,
            )
            elapsed = time.perf_counter() - training_started
            timing = {
                "completed_epochs": epoch + 1,
                "elapsed_seconds_this_process": elapsed,
                "mean_epoch_seconds_this_process": sum(epoch_times) / len(epoch_times),
                "estimated_16_epoch_seconds": (
                    elapsed * config.optimizer.epochs / len(epoch_times)
                ),
            }
            (output_dir / "training_time.json").write_text(
                json.dumps(timing, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        if distributed:
            dist.barrier()

    if distributed:
        dist.destroy_process_group()
    return output_dir / "last.pt"


def _create_optimizer(
    model: ComposedRetrievalModel,
    config: SemanticDistillationConfig,
) -> AdamW:
    backbone_parameters = [
        parameter for parameter in model.backbone.parameters() if parameter.requires_grad
    ]
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    head_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in backbone_ids
    ]
    groups = [{"params": head_parameters, "lr": config.optimizer.learning_rate}]
    if backbone_parameters:
        groups.append(
            {
                "params": backbone_parameters,
                "lr": config.optimizer.backbone_learning_rate,
            }
        )
    return AdamW(groups, weight_decay=config.optimizer.weight_decay)


def _cosine_schedule(*, total_steps: int, warmup_steps: int):
    def schedule(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return schedule


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: (
            value.to(device, non_blocking=True)
            if isinstance(value, torch.Tensor)
            else value
        )
        for key, value in batch.items()
    }


def _unwrap(model) -> ComposedRetrievalModel:
    return model.module if isinstance(model, DistributedDataParallel) else model


def _audit_trainable_gradients(model: ComposedRetrievalModel) -> dict[str, int]:
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    missing = [name for name, parameter in trainable if parameter.grad is None]
    nonfinite = [
        name
        for name, parameter in trainable
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    if missing or nonfinite:
        raise RuntimeError(
            "invalid first-batch gradients; "
            f"missing={missing[:12]} nonfinite={nonfinite[:12]}"
        )
    return {
        "trainable_tensor_count": len(trainable),
        "trainable_parameter_count": sum(parameter.numel() for _, parameter in trainable),
        "nonzero_gradient_tensor_count": sum(
            bool(torch.count_nonzero(parameter.grad)) for _, parameter in trainable
        ),
        "missing_gradient_count": 0,
        "nonfinite_gradient_count": 0,
    }


def _setup_distributed() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    return distributed, rank, local_rank, world_size


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(_worker_id: int) -> None:
    random.seed(torch.initial_seed() % 2**32)
