from __future__ import annotations

import json
import math
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

from cir_qdiag_train.oap_config import OAPConfig
from cir_qdiag_train.oap_data import CIRROAPCollator, CIRROAPDataset
from cir_qdiag_train.oap_loss import oap_ordering_loss
from cir_qdiag_train.oap_model import (
    OAPSemanticModel,
    build_oap_model,
    load_oap_checkpoint,
    save_oap_checkpoint,
)
from cir_qdiag_train.trainer import _seed_worker, _set_seed, _setup_distributed


def train_oap(
    config: OAPConfig, *, resume_from: str | Path | None = None
) -> Path:
    distributed, rank, local_rank, world_size = _setup_distributed()
    _set_seed(config.experiment.seed + rank)
    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True

    output_dir = config.experiment.output_dir
    if rank == 0:
        if resume_from is None and (output_dir / "last.pt").exists():
            raise FileExistsError(
                f"{output_dir / 'last.pt'} already exists; pass --resume-from or use "
                "a new output directory"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "effective_config.json").write_text(
            json.dumps(config.model_dump(mode="json"), ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )

    model, processor = build_oap_model(config)
    model.to(device)
    dataset = CIRROAPDataset(
        config.data.train_manifest,
        max_negatives_per_semantic=config.data.max_negatives_per_semantic,
        seed=config.experiment.seed,
    )
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=config.experiment.seed,
            drop_last=True,
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
        persistent_workers=config.data.num_workers > 0,
        drop_last=True,
        collate_fn=CIRROAPCollator(processor),
        worker_init_fn=_seed_worker,
        generator=generator,
    )
    if not len(loader):
        raise ValueError("training loader is empty; reduce batch_size_per_gpu")

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = AdamW(
        trainable_parameters,
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
    )
    updates_per_epoch = math.ceil(
        len(loader) / config.optimizer.gradient_accumulation_steps
    )
    total_steps = updates_per_epoch * config.optimizer.epochs
    warmup_steps = int(total_steps * config.optimizer.warmup_ratio)
    scheduler = LambdaLR(
        optimizer,
        _cosine_schedule(total_steps=total_steps, warmup_steps=warmup_steps),
    )
    start_epoch = 0
    global_step = 0
    if resume_from is not None:
        checkpoint = load_oap_checkpoint(
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
        raw_model = _unwrap(model)
        selected_counts = {"object": 0, "attribute": 0, "predicate": 0}
        for record in dataset.records:
            for negative in record.negatives:
                if negative.edit_operation in {"add", "substitute"}:
                    selected_counts[negative.semantic_target] += 1
        summary = {
            "num_train_samples": len(dataset),
            "source_manifest": str(config.data.train_manifest),
            "base_checkpoint": str(config.model.base_checkpoint),
            "raw_add_substitute_by_semantic": selected_counts,
            "max_negatives_per_semantic_per_epoch": (
                config.data.max_negatives_per_semantic
            ),
            "world_size": world_size,
            "steps_per_epoch_per_rank": len(loader),
            "total_optimizer_steps": total_steps,
            **raw_model.parameter_summary(),
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
                if device.type == "cuda" and config.optimizer.mixed_precision != "no"
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
                )
                raw_model = _unwrap(model)
                loss, metrics = oap_ordering_loss(
                    model=raw_model,
                    positive_global=outputs["positive_global"],
                    positive_semantic=outputs["positive_semantic"],
                    target_global=outputs["target_global"],
                    target_semantic=outputs["target_semantic"],
                    negative_semantic=outputs["negative_semantic"],
                    negative_owner=batch["negative_owner"],
                    negative_semantic_class=batch["negative_semantic_class"],
                    temperature=config.optimizer.ordering_temperature,
                    retrieval_margin=config.optimizer.retrieval_margin,
                    patch_margin=config.optimizer.patch_margin,
                )
                scaled_loss = loss / config.optimizer.gradient_accumulation_steps
                scaler.scale(scaled_loss).backward()

            if not gradient_audited:
                audit = _audit_gradients(raw_model)
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
                trainable_parameters, config.optimizer.max_grad_norm
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            if rank == 0 and global_step % config.optimizer.log_every_steps == 0:
                payload: dict[str, Any] = {
                    "epoch": epoch,
                    "epoch_number": epoch + 1,
                    "global_step": global_step,
                    "learning_rate": scheduler.get_last_lr()[0],
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
                    "a", encoding="utf-8"
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
            save_oap_checkpoint(
                output_dir / f"checkpoint_epoch_{epoch + 1:03d}.pt",
                model=raw_model,
                optimizer=None,
                scheduler=None,
                epoch=epoch,
                global_step=global_step,
                config=config.model_dump(mode="json"),
            )
            save_oap_checkpoint(
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


def _unwrap(model: OAPSemanticModel | DistributedDataParallel) -> OAPSemanticModel:
    return model.module if isinstance(model, DistributedDataParallel) else model


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: (
            value.to(device, non_blocking=True)
            if isinstance(value, torch.Tensor)
            else value
        )
        for key, value in batch.items()
    }


def _audit_gradients(model: OAPSemanticModel) -> dict[str, Any]:
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
    frozen_with_grad = [
        name
        for name, parameter in model.base.named_parameters()
        if parameter.grad is not None
    ]
    if missing or nonfinite or frozen_with_grad:
        raise RuntimeError(
            "OAP gradient audit failed; "
            f"missing={missing[:8]}, nonfinite={nonfinite[:8]}, "
            f"frozen_base_with_grad={frozen_with_grad[:8]}"
        )
    return {
        "trainable_tensor_count": len(trainable),
        "trainable_parameter_count": sum(
            parameter.numel() for _, parameter in trainable
        ),
        "missing_gradient_count": 0,
        "nonfinite_gradient_count": 0,
        "frozen_base_gradient_count": 0,
    }


def _cosine_schedule(*, total_steps: int, warmup_steps: int):
    def schedule(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return schedule
