from __future__ import annotations

import json
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from cir_qdiag_train.config import TrainConfig
from cir_qdiag_train.data import CIRRHardNegativeDataset, CIRRTrainCollator
from cir_qdiag_train.losses import (
    composed_infonce_loss,
    typed_candidate_counterfactual_loss,
)
from cir_qdiag_train.manifest import read_manifest
from cir_qdiag_train.model import (
    ComposedRetrievalModel,
    load_checkpoint,
    save_checkpoint,
)


def train(
    config: TrainConfig,
    *,
    resume_from: str | Path | None = None,
    initialize_from: str | Path | None = None,
) -> Path:
    if resume_from is not None and initialize_from is not None:
        raise ValueError("resume_from and initialize_from are mutually exclusive")
    distributed, rank, local_rank, world_size = _setup_distributed()
    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )
    _set_seed(config.experiment.seed + rank)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True

    output_dir = config.experiment.output_dir
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "effective_config.json").write_text(
            json.dumps(config.model_dump(mode="json"), ensure_ascii=False, indent=2)
            + "\n",
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
    if initialize_from is not None:
        load_checkpoint(initialize_from, model=model)
    model.to(device)

    dataset = CIRRHardNegativeDataset(
        config.data.train_manifest,
        negative_mode=config.experiment.negative_mode,
        max_non_delete_negatives_per_sample=(
            config.data.max_non_delete_negatives_per_sample
        ),
        max_delete_negatives_per_sample=config.data.max_delete_negatives_per_sample,
        seed=config.experiment.seed,
        require_usable_negatives=config.optimizer.loss_mode == "hard_only",
        candidate_image_mode=config.data.candidate_image_mode,
        train_image_split_path=config.data.train_image_split_path,
        image_root=config.data.image_root,
        max_candidate_images_per_sample=(
            config.data.max_candidate_images_per_sample
        ),
    )
    drop_incomplete_batch = config.optimizer.loss_mode != "hard_only"
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=config.experiment.seed,
            drop_last=drop_incomplete_batch,
        )
        if distributed
        else None
    )
    generator = torch.Generator()
    generator.manual_seed(config.experiment.seed + rank)
    loader = DataLoader(
        dataset,
        batch_size=config.optimizer.batch_size_per_gpu,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=config.data.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=drop_incomplete_batch,
        collate_fn=CIRRTrainCollator(processor),
        worker_init_fn=_seed_worker,
        generator=generator,
    )
    if not loader:
        raise ValueError("training loader is empty; reduce batch_size_per_gpu")

    optimizer = _create_optimizer(model, config)
    update_steps_per_epoch = math.ceil(
        len(loader) / config.optimizer.gradient_accumulation_steps
    )
    total_steps = update_steps_per_epoch * config.optimizer.epochs
    warmup_steps = int(total_steps * config.optimizer.warmup_ratio)
    scheduler = LambdaLR(
        optimizer,
        _cosine_schedule(total_steps=total_steps, warmup_steps=warmup_steps),
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
        manifest_records = read_manifest(config.data.train_manifest)
        _write_run_summary(
            output_dir,
            records=manifest_records,
            negative_mode=config.experiment.negative_mode,
            world_size=world_size,
            steps_per_epoch=len(loader),
            total_steps=total_steps,
            parameter_summary=_unwrap(model).parameter_summary(),
            loss_mode=config.optimizer.loss_mode,
            candidate_image_mode=config.data.candidate_image_mode,
            max_candidate_images_per_sample=(
                config.data.max_candidate_images_per_sample
            ),
            candidate_negative_weight=config.optimizer.candidate_negative_weight,
            counterfactual_square_weight=(
                config.optimizer.counterfactual_square_weight
            ),
            hard_negative_margin=config.optimizer.hard_negative_margin,
            initialize_from=initialize_from,
            resume_from=resume_from,
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
    for epoch in range(start_epoch, config.optimizer.epochs):
        dataset.set_epoch(epoch)
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
                if device.type == "cuda"
                and config.optimizer.mixed_precision != "no"
                else nullcontext()
            )
            with sync_context, amp_context:
                outputs = model(
                    reference_pixel_values=batch["reference_pixel_values"],
                    target_pixel_values=batch["target_pixel_values"],
                    positive_input_ids=batch["positive_input_ids"],
                    positive_attention_mask=batch["positive_attention_mask"],
                    negative_input_ids=batch["negative_input_ids"],
                    negative_attention_mask=batch["negative_attention_mask"],
                    negative_owner=batch["negative_owner"],
                    candidate_pixel_values=batch["candidate_pixel_values"],
                )
                loss, metrics = composed_infonce_loss(
                    positive_queries=outputs["positive_queries"],
                    target_features=outputs["target_features"],
                    negative_queries=outputs["negative_queries"],
                    negative_owner=batch["negative_owner"],
                    scale=outputs["scale"],
                    hard_negative_weight=config.optimizer.hard_negative_weight,
                    hard_negative_margin=config.optimizer.hard_negative_margin,
                    loss_mode=config.optimizer.loss_mode,
                )
                candidate_loss, candidate_metrics = (
                    typed_candidate_counterfactual_loss(
                        positive_queries=outputs["positive_queries"],
                        target_features=outputs["target_features"],
                        negative_queries=outputs["negative_queries"],
                        negative_owner=batch["negative_owner"],
                        negative_semantic_class=batch[
                            "negative_semantic_class"
                        ],
                        candidate_features=outputs["candidate_features"],
                        candidate_owner=batch["candidate_owner"],
                        scale=outputs["scale"],
                        candidate_negative_weight=(
                            config.optimizer.candidate_negative_weight
                        ),
                        counterfactual_square_weight=(
                            config.optimizer.counterfactual_square_weight
                        ),
                        candidate_margin=config.optimizer.candidate_margin,
                        counterfactual_margin=(
                            config.optimizer.counterfactual_margin
                        ),
                        mining_min_advantage=(
                            config.optimizer.counterfactual_mining_min_advantage
                        ),
                    )
                )
                loss = loss + candidate_loss
                metrics.update(candidate_metrics)
                metrics["loss"] = loss.detach()
                loss = loss / config.optimizer.gradient_accumulation_steps
                scaler.scale(loss).backward()

            if not gradient_audited:
                audit = _audit_trainable_gradients(_unwrap(model))
                if rank == 0:
                    write_path = output_dir / "gradient_audit.json"
                    write_path.write_text(
                        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    print(json.dumps({"gradient_audit": audit}, ensure_ascii=False))
                gradient_audited = True

            for key, value in metrics.items():
                running[key] = running.get(key, 0.0) + float(value.item())
            running_count += 1
            if not sync_step:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                _unwrap(model).parameters(),
                config.optimizer.max_grad_norm,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            if rank == 0 and global_step % config.optimizer.log_every_steps == 0:
                learning_rates = scheduler.get_last_lr()
                payload: dict[str, Any] = {
                    "epoch": epoch,
                    "epoch_number": epoch + 1,
                    "global_step": global_step,
                    "learning_rate": learning_rates[0],
                    "head_learning_rate": learning_rates[0],
                    "backbone_learning_rate": (
                        learning_rates[1] if len(learning_rates) > 1 else None
                    ),
                    "temperature": float(
                        (1.0 / _unwrap(model).scale()).detach().cpu().item()
                    ),
                }
                payload.update(
                    {
                        key: round(value / running_count, 6)
                        for key, value in running.items()
                    }
                )
                payload_text = json.dumps(payload, ensure_ascii=False)
                print(payload_text, flush=True)
                with (output_dir / "train_metrics.jsonl").open(
                    "a",
                    encoding="utf-8",
                ) as handle:
                    handle.write(payload_text + "\n")
                running.clear()
                running_count = 0

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
            )
            save_checkpoint(
                output_dir / "last.pt",
                model=raw_model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                config=config.model_dump(mode="json"),
            )
        if distributed:
            dist.barrier()

    if distributed:
        dist.destroy_process_group()
    return output_dir / "last.pt"


def _create_optimizer(
    model: ComposedRetrievalModel,
    config: TrainConfig,
) -> AdamW:
    backbone_parameters = [
        parameter
        for parameter in model.backbone.parameters()
        if parameter.requires_grad
    ]
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    head_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in backbone_ids
    ]
    parameter_groups = [
        {"params": head_parameters, "lr": config.optimizer.learning_rate},
    ]
    if backbone_parameters:
        parameter_groups.append(
            {
                "params": backbone_parameters,
                "lr": config.optimizer.backbone_learning_rate,
            }
        )
    return AdamW(
        parameter_groups,
        weight_decay=config.optimizer.weight_decay,
    )


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


def _audit_trainable_gradients(model: ComposedRetrievalModel) -> dict[str, Any]:
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    missing = [name for name, parameter in trainable if parameter.grad is None]
    if missing:
        raise RuntimeError(
            "trainable parameters received no gradient on the first batch; "
            f"count={len(missing)} first={missing[:12]}"
        )
    nonfinite = [
        name
        for name, parameter in trainable
        if not torch.isfinite(parameter.grad).all()
    ]
    if nonfinite:
        raise RuntimeError(
            "non-finite gradients on the first batch; "
            f"count={len(nonfinite)} first={nonfinite[:12]}"
        )
    nonzero = sum(bool(torch.count_nonzero(parameter.grad)) for _, parameter in trainable)
    backbone_trainable = [
        (name, parameter)
        for name, parameter in model.backbone.named_parameters()
        if parameter.requires_grad
    ]
    return {
        "trainable_tensor_count": len(trainable),
        "trainable_parameter_count": sum(parameter.numel() for _, parameter in trainable),
        "trainable_tensors_with_nonzero_gradient": nonzero,
        "trainable_backbone_tensor_count": len(backbone_trainable),
        "trainable_backbone_parameter_count": sum(
            parameter.numel() for _, parameter in backbone_trainable
        ),
        "missing_gradient_count": 0,
        "nonfinite_gradient_count": 0,
    }


def _setup_distributed() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    return distributed, rank, local_rank, world_size


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)


def _write_run_summary(
    output_dir: Path,
    *,
    records,
    negative_mode: str,
    world_size: int,
    steps_per_epoch: int,
    total_steps: int,
    parameter_summary: dict[str, Any],
    loss_mode: str,
    candidate_image_mode: str,
    max_candidate_images_per_sample: int,
    candidate_negative_weight: float,
    counterfactual_square_weight: float,
    hard_negative_margin: float,
    initialize_from: str | Path | None,
    resume_from: str | Path | None,
) -> None:
    operation_counts = {"add": 0, "delete": 0, "substitute": 0}
    samples_with_usable_negatives = 0
    for record in records:
        if negative_mode == "none":
            usable = []
        else:
            usable = [
                item
                for item in record.negatives
                if negative_mode == "add_substitute_delete"
                or item.edit_operation != "delete"
            ]
        samples_with_usable_negatives += bool(usable)
        for item in usable:
            operation_counts[item.edit_operation] += 1
    payload = {
        "num_train_samples": len(records),
        "samples_with_usable_negatives": samples_with_usable_negatives,
        "negative_mode": negative_mode,
        "loss_mode": loss_mode,
        "candidate_image_mode": candidate_image_mode,
        "max_candidate_images_per_sample": max_candidate_images_per_sample,
        "candidate_negative_weight": candidate_negative_weight,
        "counterfactual_square_weight": counterfactual_square_weight,
        "hard_negative_margin": hard_negative_margin,
        "uses_hard_negatives": negative_mode != "none",
        "initialize_from": (
            None if initialize_from is None else str(Path(initialize_from).resolve())
        ),
        "resume_from": None if resume_from is None else str(Path(resume_from).resolve()),
        "usable_negatives_by_operation": operation_counts,
        "world_size": world_size,
        "steps_per_epoch_per_rank": steps_per_epoch,
        "total_optimizer_steps": total_steps,
        **parameter_summary,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
