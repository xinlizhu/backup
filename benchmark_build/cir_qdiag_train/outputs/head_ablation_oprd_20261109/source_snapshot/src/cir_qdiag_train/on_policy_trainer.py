from __future__ import annotations

import json
import gc
import math
import os
import random
import time
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from cir_qdiag_train.model import ComposedRetrievalModel, load_checkpoint, save_checkpoint
from cir_qdiag_train.on_policy_config import OnPolicyDistillationConfig
from cir_qdiag_train.natural_rank_loss import natural_candidate_rank_loss
from cir_qdiag_train.on_policy_data import (
    InMemoryReservoirCollator,
    InMemoryReservoirDataset,
    OnPolicyTrainingCollator,
    OnPolicyTrainingDataset,
    load_cirr_positive_records,
    load_image_map,
)
from cir_qdiag_train.on_policy_loss import (
    CausalPolicyRollouts,
    candidate_infonce_loss,
    causal_prefix_anchor_kl,
    causal_group_policy_loss,
    centered_similarity_mse_loss,
    listwise_group_relative_loss,
    merge_mixed_candidate_features,
    on_policy_centered_distillation_loss,
    reward_tilted_reverse_opd_loss,
)
from cir_qdiag_train.on_policy_mining import (
    build_student_reservoir_in_memory,
    refresh_on_policy_candidates,
    write_json_atomic,
)
from cir_qdiag_train.pixel_cache import (
    GalleryPixelCache,
    build_gallery_pixel_cache,
    iter_prefetched_pixel_chunks,
)


@dataclass
class _AdaptiveAnchorKLController:
    enabled: bool
    target: float
    dual_lr: float
    ema_decay: float
    update_interval: int
    beta_min: float
    beta_max: float
    beta: float
    ema: float | None = None
    observations: int = 0

    def reset_epoch_average(self) -> None:
        self.ema = None
        self.observations = 0

    def observe(self, value: float) -> None:
        if not self.enabled:
            return
        self.ema = (
            value
            if self.ema is None
            else self.ema_decay * self.ema + (1.0 - self.ema_decay) * value
        )
        self.observations += 1
        if self.observations % self.update_interval != 0:
            return
        log_beta = math.log(self.beta)
        log_beta += self.dual_lr * (self.ema / self.target - 1.0)
        self.beta = min(
            self.beta_max,
            max(self.beta_min, math.exp(log_beta)),
        )

    def state_dict(self) -> dict[str, float | None]:
        return {
            "beta": self.beta,
            "ema": self.ema,
            "observations": self.observations,
        }


def train_on_policy_distillation(
    config: OnPolicyDistillationConfig,
    *,
    resume_from: str | Path | None = None,
) -> Path:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 1:
        raise ValueError(
            "on-policy full-gallery mining currently supports one process/GPU; "
            "run with CUDA_VISIBLE_DEVICES=0 and no multi-process torchrun"
        )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _set_seed(config.experiment.seed)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True

    output_dir = config.experiment.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        output_dir / "effective_config.json",
        config.model_dump(mode="json"),
    )
    image_ids, image_paths = load_image_map(
        config.data.train_image_split_path, config.data.image_root
    )
    records = load_cirr_positive_records(
        config.data.train_annotation_path, image_ids, image_paths
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
    initial_checkpoint_metadata: dict[str, Any] | None = None
    if resume_from is None and config.experiment.initial_checkpoint is not None:
        if not config.experiment.initial_checkpoint.is_file():
            raise FileNotFoundError(config.experiment.initial_checkpoint)
        initial_checkpoint_metadata = load_checkpoint(
            config.experiment.initial_checkpoint,
            model=model,
        )
    gallery_pixel_cache: GalleryPixelCache | None = None
    if config.mining.pixel_cache_enabled:
        pixel_cache_dtype = {
            "float16": torch.float16,
            "float32": torch.float32,
        }[config.mining.pixel_cache_dtype]
        gallery_pixel_cache = build_gallery_pixel_cache(
            image_paths=image_paths,
            processor=processor,
            dtype=pixel_cache_dtype,
            batch_size=config.mining.pixel_cache_build_batch_size,
            workers=config.mining.pixel_cache_workers,
            prefetch_factor=config.mining.pixel_cache_build_prefetch_factor,
        )
    model.to(device)
    optimizer = _create_optimizer(model, config)
    data_steps_per_epoch = math.ceil(
        math.ceil(len(records) / config.optimizer.batch_size_per_gpu)
        / config.optimizer.gradient_accumulation_steps
    )
    policy_updates_per_batch = (
        config.objective.policy_update_epochs
        if config.objective.is_causal_policy_loss
        else 1
    )
    steps_per_epoch = data_steps_per_epoch * policy_updates_per_batch
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
    resumed_anchor_kl_state: dict[str, Any] = {}
    if resume_from is not None:
        checkpoint = load_checkpoint(
            resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        resumed_anchor_kl_state = dict(
            checkpoint.get("metrics", {}).get("anchor_kl_controller", {})
        )

    anchor_kl_controller = _AdaptiveAnchorKLController(
        enabled=config.objective.anchor_kl_enabled,
        target=config.objective.anchor_kl_target,
        dual_lr=config.objective.anchor_kl_dual_lr,
        ema_decay=config.objective.anchor_kl_ema_decay,
        update_interval=config.objective.anchor_kl_update_interval,
        beta_min=config.objective.anchor_kl_beta_min,
        beta_max=config.objective.anchor_kl_beta_max,
        beta=float(
            resumed_anchor_kl_state.get(
                "beta", config.objective.anchor_kl_beta_initial
            )
        ),
        ema=(
            None
            if resumed_anchor_kl_state.get("ema") is None
            else float(resumed_anchor_kl_state["ema"])
        ),
        observations=int(resumed_anchor_kl_state.get("observations", 0)),
    )

    write_json_atomic(
        output_dir / "run_summary.json",
        {
            "algorithm": {
                "candidate_infonce": "candidate InfoNCE",
                "rank_kd": "candidate-set listwise rank distillation",
                "centered_mse": "centered raw-similarity mean squared error",
                "listwise_group_relative": (
                    "target-NDCG listwise group-relative policy gradient"
                ),
                "causal_reinforce": (
                    "target-NDCG causal-prefix group-relative REINFORCE"
                ),
                "causal_grpo": (
                    "target-NDCG causal-prefix position-level GRPO"
                ),
                "causal_gspo": (
                    "target-NDCG causal-prefix sequence-level GSPO"
                ),
                "reward_tilted_reverse_opd": (
                    "NDCG-tilted Teacher rollout direct reverse-KL OPD"
                ),
                "reward_tilted_reverse_gspo": (
                    "NDCG-tilted Teacher reverse-KL causal-prefix GSPO"
                ),
            }[config.objective.loss_type],
            "student_initialization": (
                f"checkpoint: {config.experiment.initial_checkpoint}"
                if config.experiment.initial_checkpoint is not None
                else f"original pretrained {config.model.variant}: "
                f"{config.model.pretrained_model_name_or_path}"
            ),
            "student_checkpoint_initialization": (
                None
                if config.experiment.initial_checkpoint is None
                else str(config.experiment.initial_checkpoint)
            ),
            "student_checkpoint_initial_epoch": (
                None
                if initial_checkpoint_metadata is None
                else int(initial_checkpoint_metadata.get("epoch", -1)) + 1
            ),
            "teacher_backend": config.teacher.backend,
            "teacher_cache": (
                None
                if config.data.teacher_cache_dir is None
                else str(config.data.teacher_cache_dir)
            ),
            "teacher_endpoint_url": config.teacher.endpoint_url,
            "teacher_model_name": config.teacher.model_name,
            "teacher_api_format": config.teacher.api_format,
            "teacher_instruction": config.teacher.instruction,
            "teacher_image_transport": config.teacher.image_transport,
            "uses_qdiag_negative_queries": (
                config.mining.include_negative_queries_as_candidates
            ),
            "uses_infonce": config.objective.loss_type == "candidate_infonce",
            "uses_teacher_filter": config.objective.uses_teacher_filter,
            "loss_type": config.objective.loss_type,
            "teacher_similarity_target": (
                "raw similarities up to a per-query additive offset"
                if config.objective.loss_type == "centered_mse"
                else "NDCG-tilted Teacher rollout probability distribution"
                if config.objective.loss_type in {
                    "reward_tilted_reverse_opd",
                    "reward_tilted_reverse_gspo",
                }
                else "temperature-softmax distribution"
                if config.objective.loss_type == "rank_kd"
                else "none; gold target NDCG reward"
                if config.objective.is_policy_loss
                else "filter only; target is candidate zero"
            ),
            "student_similarity_input": (
                "raw similarities; no softmax or temperature"
                if config.objective.loss_type == "centered_mse"
                else f"similarities divided by T={config.objective.rank_temperature}"
            ),
            "train_annotation_count": len(records),
            "train_gallery_count": len(image_ids),
            "candidate_count_per_query": config.mining.top_k,
            "epoch_reservoir_candidate_count": config.mining.reservoir_top_k,
            "live_reservoir_selection": config.mining.live_reservoir_selection,
            "live_selection_image_batch_size": (
                config.mining.live_selection_image_batch_size
                if config.mining.live_reservoir_selection
                else None
            ),
            "gallery_pixel_cache_enabled": gallery_pixel_cache is not None,
            "gallery_pixel_cache_dtype": (
                None
                if gallery_pixel_cache is None
                else str(gallery_pixel_cache.pixel_values.dtype)
            ),
            "gallery_pixel_cache_size_gib": (
                None
                if gallery_pixel_cache is None
                else gallery_pixel_cache.size_bytes / (1024 ** 3)
            ),
            "gallery_pixel_cache_build_seconds": (
                None
                if gallery_pixel_cache is None
                else gallery_pixel_cache.build_seconds
            ),
            "gallery_pixel_cache_workers": config.mining.pixel_cache_workers,
            "gallery_pixel_cache_build_prefetch_factor": (
                config.mining.pixel_cache_build_prefetch_factor
            ),
            "gallery_pixel_cache_transfer_prefetch_batches": (
                config.mining.pixel_cache_transfer_prefetch_batches
            ),
            "persistent_candidate_cache": config.mining.persist_candidate_cache,
            "target_forced_into_candidate_set": config.mining.force_target_insertion,
            "candidate_layout": (
                "natural Student Top-K without target insertion"
                if not config.mining.force_target_insertion else
                "target plus owned composed q- candidates plus current hard images"
                if config.mining.include_negative_queries_as_candidates
                else "target plus random non-target images"
                if config.mining.candidate_source == "random"
                else "target plus current top-(K-1) non-target images"
            ),
            "candidate_source": config.mining.candidate_source,
            "candidate_refresh_mode": config.mining.refresh_mode,
            "candidate_selection_features": (
                f"{config.mining.candidate_source} retrieval policy"
            ),
            "training_candidate_features": (
                "all reference/candidate images re-encoded by the current Student "
                "after each minibatch is selected"
            ),
            "stale_candidate_features_used_in_loss": False,
            "candidate_refresh": config.mining.refresh_mode,
            "rank_weight": config.objective.rank_weight,
            "rank_kl_direction": config.objective.rank_kl_direction,
            "reward_tilt_beta": (
                config.objective.reward_tilt_beta
                if config.objective.loss_type in {
                    "reward_tilted_reverse_opd",
                    "reward_tilted_reverse_gspo",
                }
                else None
            ),
            "rollout_count": (
                config.objective.rollout_count
                if config.objective.is_policy_loss
                else None
            ),
            "rollout_top_k": (
                config.objective.rollout_top_k
                if config.objective.is_policy_loss
                else None
            ),
            "rl_reward": (
                "single-positive target NDCG at sampled rollout_top_k"
                if config.objective.is_policy_loss
                else None
            ),
            "rl_baseline": (
                "within-query group reward mean and standard deviation"
                if config.objective.is_policy_loss
                else None
            ),
            "rl_causal_prefix": config.objective.is_causal_policy_loss,
            "rl_policy_update_epochs": (
                config.objective.policy_update_epochs
                if config.objective.is_causal_policy_loss
                else None
            ),
            "rl_policy_clip_epsilon": (
                config.objective.policy_clip_epsilon
                if config.objective.is_causal_policy_loss
                else None
            ),
            "rl_deterministic_policy_dropout": (
                config.objective.deterministic_policy_dropout
                if config.objective.is_causal_policy_loss
                else None
            ),
            "anchor_policy": (
                "epoch-start Student distribution on the selected live candidates"
                if config.objective.anchor_kl_enabled
                else None
            ),
            "anchor_kl_enabled": config.objective.anchor_kl_enabled,
            "anchor_kl_direction": (
                "KL(anchor || current)"
                if config.objective.anchor_kl_enabled
                else None
            ),
            "anchor_kl_target": (
                config.objective.anchor_kl_target
                if config.objective.anchor_kl_enabled
                else None
            ),
            "anchor_kl_beta_initial": (
                config.objective.anchor_kl_beta_initial
                if config.objective.anchor_kl_enabled
                else None
            ),
            "centered_response_weight": config.objective.centered_response_weight,
            "teacher_target_max_rank": (
                1
                if config.objective.require_teacher_target_top1
                else config.objective.teacher_target_max_rank
            ),
            "optimizer_steps_per_epoch": steps_per_epoch,
            "data_batches_per_epoch": data_steps_per_epoch,
            "total_optimizer_steps": total_steps,
            "live_image_batch_size": config.optimizer.live_image_batch_size,
            "data_loader_workers": config.data.num_workers,
            "data_loader_prefetch_factor": (
                config.data.prefetch_factor if config.data.num_workers > 0 else None
            ),
            "max_prefetched_batches": (
                config.data.num_workers * config.data.prefetch_factor
                if config.data.num_workers > 0
                else 0
            ),
            **model.parameter_summary(),
        },
    )

    use_fp16 = config.optimizer.mixed_precision == "fp16" and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    training_started = time.perf_counter()
    epoch_timings: list[dict[str, float | int]] = []
    gradient_audited = (output_dir / "gradient_audit.json").is_file()
    for epoch_index in range(start_epoch, config.optimizer.epochs):
        epoch_number = epoch_index + 1
        epoch_started = time.perf_counter()
        mining_started = time.perf_counter()
        mining_epoch = (
            1 if config.mining.refresh_mode == "once" else epoch_number
        )
        mining_global_step = (
            0 if config.mining.refresh_mode == "once" else global_step
        )
        reservoir = None
        if config.mining.live_reservoir_selection:
            reservoir = build_student_reservoir_in_memory(
                config=config,
                model=model,
                processor=processor,
                records=records,
                image_ids=image_ids,
                image_paths=image_paths,
                epoch=mining_epoch,
                global_step=mining_global_step,
                device=device,
                pixel_cache=gallery_pixel_cache,
            )
            mining_metadata = reservoir.metadata
            dataset = InMemoryReservoirDataset(
                records=records,
                image_ids=image_ids,
                candidate_indices=reservoir.candidate_indices,
                anchor_scores=reservoir.anchor_scores,
                teacher_scores=reservoir.teacher_scores,
                target_indices=reservoir.target_indices,
                raw_target_in_reservoir=reservoir.raw_target_in_reservoir,
            )
            collate_fn = InMemoryReservoirCollator(processor)
        else:
            static_metadata = output_dir / "on_policy" / "epoch_001" / "metadata.json"
            if (
                config.mining.refresh_mode == "once"
                and start_epoch > 0
                and not static_metadata.is_file()
            ):
                raise RuntimeError(
                    "cannot resume static-candidate training without its epoch-1 cache"
                )
            mining_dir, mining_metadata = refresh_on_policy_candidates(
                config=config,
                model=model,
                processor=processor,
                records=records,
                image_ids=image_ids,
                image_paths=image_paths,
                epoch=mining_epoch,
                global_step=mining_global_step,
                device=device,
            )
            dataset = OnPolicyTrainingDataset(
                config,
                records,
                image_ids,
                image_paths,
                mining_dir,
                expected_epoch=mining_epoch,
            )
            collate_fn = OnPolicyTrainingCollator(processor, image_paths)
        mining_seconds = time.perf_counter() - mining_started
        loader = DataLoader(
            dataset,
            batch_size=config.optimizer.batch_size_per_gpu,
            shuffle=True,
            num_workers=config.data.num_workers,
            prefetch_factor=(
                config.data.prefetch_factor if config.data.num_workers > 0 else None
            ),
            pin_memory=device.type == "cuda",
            drop_last=False,
            persistent_workers=False,
            collate_fn=collate_fn,
            worker_init_fn=_seed_worker,
            generator=torch.Generator().manual_seed(config.experiment.seed + epoch_index),
        )
        model.train()
        anchor_kl_controller.reset_epoch_average()
        disabled_dropout_count = 0
        if (
            config.objective.is_causal_policy_loss
            and config.objective.deterministic_policy_dropout
        ):
            disabled_dropout_count = _disable_dropout_randomness(model)
        optimizer.zero_grad(set_to_none=True)
        running: dict[str, float] = {}
        running_count = 0
        training_phase_started = time.perf_counter()
        for micro_step, batch in enumerate(loader):
            batch = _move_batch(batch, device)
            screening_metrics: dict[str, torch.Tensor] = {}
            if config.mining.live_reservoir_selection:
                screening_started = time.perf_counter()
                # Rank candidate selection follows deterministic retrieval inference.
                if config.objective.loss_type == "rank_kd":
                    model.eval()
                selected = _select_live_topk_from_reservoir(
                    model=model,
                    processor=processor,
                    batch=batch,
                    image_paths=image_paths,
                    candidate_count=config.mining.top_k,
                    image_batch_size=(
                        config.mining.live_selection_image_batch_size
                    ),
                    mixed_precision=config.optimizer.mixed_precision,
                    image_loader_workers=config.data.num_workers,
                    device=device,
                    pixel_cache=gallery_pixel_cache,
                    transfer_prefetch_batches=(
                        config.mining.pixel_cache_transfer_prefetch_batches
                    ),
                    pin_memory=config.mining.pixel_cache_pin_memory,
                )
                screening_seconds = time.perf_counter() - screening_started
                if config.objective.loss_type == "rank_kd":
                    model.train()
                materialization_started = time.perf_counter()
                batch = _materialize_selected_candidate_batch(
                    processor=processor,
                    source_batch=batch,
                    selected_candidate_indices=selected[
                        "selected_candidate_indices"
                    ],
                    selected_anchor_scores=selected.get(
                        "selected_anchor_scores"
                    ),
                    raw_target_in_topk=selected["raw_target_in_topk"],
                    selected_teacher_scores=selected.get("selected_teacher_scores"),
                    image_paths=image_paths,
                    preprocess_batch_size=config.optimizer.live_image_batch_size,
                    image_loader_workers=config.data.num_workers,
                    device=device,
                    pixel_cache=gallery_pixel_cache,
                    transfer_prefetch_batches=(
                        config.mining.pixel_cache_transfer_prefetch_batches
                    ),
                    pin_memory=config.mining.pixel_cache_pin_memory,
                )
                screening_metrics = selected["metrics"]
                screening_metrics["live_reservoir_screening_seconds"] = (
                    batch["unique_pixel_values"].new_tensor(screening_seconds)
                )
                screening_metrics[
                    "selected_candidate_materialization_seconds"
                ] = batch["unique_pixel_values"].new_tensor(
                    time.perf_counter() - materialization_started
                )
            if config.objective.is_causal_policy_loss:
                policy_algorithm = {
                    "causal_reinforce": "reinforce",
                    "causal_grpo": "grpo",
                    "causal_gspo": "gspo",
                    "reward_tilted_reverse_gspo": "gspo",
                    "reward_tilted_reverse_opd": "gspo",
                }[config.objective.loss_type]
                fixed_rollouts: CausalPolicyRollouts | None = None
                update_losses: list[torch.Tensor] = []
                first_update_metrics: dict[str, torch.Tensor] | None = None
                metrics: dict[str, torch.Tensor] | None = None
                for policy_update_index in range(
                    config.objective.policy_update_epochs
                ):
                    optimizer.zero_grad(set_to_none=True)
                    with _autocast(device, config.optimizer.mixed_precision):
                        student_scores = _compute_live_student_scores(
                            model=model,
                            batch=batch,
                            live_image_batch_size=(
                                config.optimizer.live_image_batch_size
                            ),
                        )
                        if (
                            config.objective.loss_type
                            == "reward_tilted_reverse_opd"
                        ):
                            policy_loss, metrics = reward_tilted_reverse_opd_loss(
                                student_scores=student_scores,
                                teacher_scores=batch["teacher_scores"].float(),
                                raw_target_in_topk=batch["raw_target_in_topk"],
                                temperature=config.objective.rank_temperature,
                                rollout_count=config.objective.rollout_count,
                                rollout_top_k=config.objective.rollout_top_k,
                                reward_tilt_beta=(
                                    config.objective.reward_tilt_beta
                                ),
                                require_teacher_target_top1=(
                                    config.objective.require_teacher_target_top1
                                ),
                                teacher_target_max_rank=(
                                    config.objective.teacher_target_max_rank
                                ),
                            )
                        else:
                            policy_loss, fixed_rollouts, metrics = causal_group_policy_loss(
                                student_scores=student_scores,
                                raw_target_in_topk=batch["raw_target_in_topk"],
                                temperature=config.objective.rank_temperature,
                                rollout_count=config.objective.rollout_count,
                                rollout_top_k=config.objective.rollout_top_k,
                                algorithm=policy_algorithm,
                                clip_epsilon=config.objective.policy_clip_epsilon,
                                rollouts=fixed_rollouts,
                                teacher_scores=(
                                    batch["teacher_scores"].float()
                                    if config.objective.loss_type
                                    == "reward_tilted_reverse_gspo"
                                    else None
                                ),
                                reward_tilt_beta=(
                                    config.objective.reward_tilt_beta
                                    if config.objective.loss_type
                                    == "reward_tilted_reverse_gspo"
                                    else None
                                ),
                            )
                        loss = policy_loss
                        if config.objective.anchor_kl_enabled:
                            anchor_kl, anchor_metrics = causal_prefix_anchor_kl(
                                student_scores=student_scores,
                                anchor_scores=batch["anchor_scores"],
                                rollouts=fixed_rollouts,
                                temperature=config.objective.rank_temperature,
                            )
                            beta_used = anchor_kl_controller.beta
                            loss = loss + beta_used * anchor_kl
                            metrics = dict(metrics)
                            metrics.update(anchor_metrics)
                            metrics["anchor_kl_beta_used"] = (
                                student_scores.new_tensor(beta_used)
                            )
                            metrics["anchor_kl_target"] = (
                                student_scores.new_tensor(
                                    config.objective.anchor_kl_target
                                )
                            )
                        else:
                            anchor_kl = None
                        scaler.scale(loss).backward()
                    if first_update_metrics is None:
                        first_update_metrics = metrics
                    update_losses.append(loss.detach())
                    if not gradient_audited:
                        write_json_atomic(
                            output_dir / "gradient_audit.json",
                            _audit_trainable_gradients(model),
                        )
                        gradient_audited = True
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optimizer.max_grad_norm
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    global_step += 1
                    if anchor_kl is not None:
                        anchor_kl_controller.observe(float(anchor_kl.detach().item()))
                        metrics["anchor_kl_beta_after_update"] = (
                            student_scores.new_tensor(anchor_kl_controller.beta)
                        )
                        metrics["anchor_kl_ema"] = student_scores.new_tensor(
                            0.0
                            if anchor_kl_controller.ema is None
                            else anchor_kl_controller.ema
                        )

                assert metrics is not None and first_update_metrics is not None
                metrics = dict(metrics)
                metrics["loss"] = torch.stack(update_losses).mean()
                metrics["first_policy_update_loss"] = update_losses[0]
                metrics["last_policy_update_loss"] = update_losses[-1]
                metrics["initial_policy_entropy"] = first_update_metrics[
                    "policy_entropy"
                ]
                metrics["initial_policy_target_probability"] = (
                    first_update_metrics["policy_target_probability"]
                )
                metrics["policy_update_epochs"] = student_scores.new_tensor(
                    float(config.objective.policy_update_epochs)
                )
                metrics["rollout_group_reused"] = student_scores.new_tensor(
                    float(config.objective.policy_update_epochs > 1)
                )
                metrics["disabled_dropout_module_count"] = (
                    student_scores.new_tensor(float(disabled_dropout_count))
                )
                metrics.update(screening_metrics)
                metrics["live_unique_images_per_batch"] = (
                    batch["unique_image_count"].float().detach()
                )
                metrics["negative_query_candidates_per_batch"] = (
                    batch["candidate_is_negative_query"]
                    .sum()
                    .float()
                    .detach()
                )
                for key, value in metrics.items():
                    running[key] = running.get(key, 0.0) + float(value.item())
                running_count += 1
                if global_step % config.optimizer.log_every_steps == 0:
                    _write_training_log(
                        output_dir=output_dir,
                        epoch_number=epoch_number,
                        global_step=global_step,
                        scheduler=scheduler,
                        mining_metadata=mining_metadata,
                        config=config,
                        running=running,
                        running_count=running_count,
                    )
                    running.clear()
                    running_count = 0
                continue

            sync_step = (
                (micro_step + 1) % config.optimizer.gradient_accumulation_steps == 0
                or micro_step + 1 == len(loader)
            )
            amp_context = _autocast(device, config.optimizer.mixed_precision)
            with amp_context:
                student_scores = _compute_live_student_scores(
                    model=model,
                    batch=batch,
                    live_image_batch_size=config.optimizer.live_image_batch_size,
                )
                if not config.mining.force_target_insertion:
                    loss, metrics = natural_candidate_rank_loss(
                        student_scores=student_scores,
                        teacher_scores=batch["teacher_scores"].float(),
                        target_positions=batch["target_position"],
                        teacher_filter_target_ranks=batch["teacher_filter_target_rank"],
                        temperature=config.objective.rank_temperature,
                        direction=config.objective.rank_kl_direction,
                        max_teacher_rank=(1 if config.objective.require_teacher_target_top1
                                          else config.objective.teacher_target_max_rank),
                        rank_weight=config.objective.rank_weight,
                    )
                elif config.objective.loss_type == "listwise_group_relative":
                    loss, metrics = listwise_group_relative_loss(
                        student_scores=student_scores,
                        raw_target_in_topk=batch["raw_target_in_topk"],
                        temperature=config.objective.rank_temperature,
                        rollout_count=config.objective.rollout_count,
                        rollout_top_k=config.objective.rollout_top_k,
                    )
                elif config.objective.loss_type == "candidate_infonce":
                    loss, metrics = candidate_infonce_loss(
                        student_scores=student_scores,
                        raw_target_in_topk=batch["raw_target_in_topk"],
                        temperature=config.objective.rank_temperature,
                        teacher_scores=(
                            batch["teacher_scores"].float()
                            if config.objective.needs_teacher_scores
                            else None
                        ),
                        require_teacher_target_top1=(
                            config.objective.require_teacher_target_top1
                        ),
                        teacher_target_max_rank=(
                            config.objective.teacher_target_max_rank
                        ),
                    )
                elif config.objective.loss_type == "centered_mse":
                    loss, metrics = centered_similarity_mse_loss(
                        student_scores=student_scores,
                        teacher_scores=batch["teacher_scores"].float(),
                        raw_target_in_topk=batch["raw_target_in_topk"],
                        require_teacher_target_top1=(
                            config.objective.require_teacher_target_top1
                        ),
                        teacher_target_max_rank=(
                            config.objective.teacher_target_max_rank
                        ),
                    )
                else:
                    loss, metrics = on_policy_centered_distillation_loss(
                        student_scores=student_scores,
                        teacher_scores=batch["teacher_scores"].float(),
                        teacher_target_top1=batch["teacher_target_top1"],
                        raw_target_in_topk=batch["raw_target_in_topk"],
                        rank_weight=config.objective.rank_weight,
                        centered_response_weight=(
                            config.objective.centered_response_weight
                        ),
                        rank_temperature=config.objective.rank_temperature,
                        rank_kl_direction=config.objective.rank_kl_direction,
                        centered_response_beta=config.objective.centered_response_beta,
                        require_teacher_target_top1=(
                            config.objective.require_teacher_target_top1
                        ),
                        teacher_target_max_rank=(
                            config.objective.teacher_target_max_rank
                        ),
                    )
                metrics["live_unique_images_per_batch"] = (
                    batch["unique_image_count"].float().detach()
                )
                metrics["negative_query_candidates_per_batch"] = (
                    batch["candidate_is_negative_query"]
                    .sum()
                    .float()
                    .detach()
                )
                scaled_loss = loss / config.optimizer.gradient_accumulation_steps
                metrics.update(screening_metrics)
                scaler.scale(scaled_loss).backward()

            if not gradient_audited:
                write_json_atomic(
                    output_dir / "gradient_audit.json",
                    _audit_trainable_gradients(model),
                )
                gradient_audited = True
            for key, value in metrics.items():
                running[key] = running.get(key, 0.0) + float(value.item())
            running_count += 1
            if not sync_step:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            if global_step % config.optimizer.log_every_steps == 0:
                _write_training_log(
                    output_dir=output_dir,
                    epoch_number=epoch_number,
                    global_step=global_step,
                    scheduler=scheduler,
                    mining_metadata=mining_metadata,
                    config=config,
                    running=running,
                    running_count=running_count,
                )
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
                "elapsed_seconds_this_process": time.perf_counter() - training_started,
                "epochs_this_process": epoch_timings,
                "anchor_kl_controller": anchor_kl_controller.state_dict(),
            },
        )
        if (
            epoch_number % config.optimizer.save_every_epochs == 0
            or epoch_number == config.optimizer.epochs
        ):
            save_checkpoint(
                output_dir / f"checkpoint_epoch_{epoch_number:03d}.pt",
                model=model,
                optimizer=None,
                scheduler=None,
                epoch=epoch_index,
                global_step=global_step,
                config=config.model_dump(mode="json"),
                metrics={
                    "anchor_kl_controller": anchor_kl_controller.state_dict()
                },
            )
            save_checkpoint(
                output_dir / "last.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch_index,
                global_step=global_step,
                config=config.model_dump(mode="json"),
                metrics={
                    "anchor_kl_controller": anchor_kl_controller.state_dict()
                },
            )
        del loader, dataset, reservoir
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if gallery_pixel_cache is not None:
        cache_size_gib = gallery_pixel_cache.size_bytes / (1024 ** 3)
        del gallery_pixel_cache
        gc.collect()
        print(
            f"[gallery pixel cache released] size_gib={cache_size_gib:.3f}",
            flush=True,
        )
    return output_dir / "last.pt"


def _create_optimizer(
    model: ComposedRetrievalModel,
    config: OnPolicyDistillationConfig,
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
    return AdamW(
        [
            {"params": head_parameters, "lr": config.optimizer.learning_rate},
            {
                "params": backbone_parameters,
                "lr": config.optimizer.backbone_learning_rate,
            },
        ],
        weight_decay=config.optimizer.weight_decay,
    )


def _cosine_schedule(*, total_steps: int, warmup_steps: int):
    def schedule(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return schedule


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def _select_live_topk_from_reservoir(
    *,
    model: ComposedRetrievalModel,
    processor,
    batch: dict[str, Any],
    image_paths: list[Path],
    candidate_count: int,
    image_batch_size: int,
    mixed_precision: str,
    image_loader_workers: int,
    device: torch.device,
    pixel_cache: GalleryPixelCache | None = None,
    transfer_prefetch_batches: int = 2,
    pin_memory: bool = True,
) -> dict[str, Any]:
    """Use the latest Student to screen an epoch-start Top-M reservoir."""

    reservoir_indices = batch["reservoir_candidate_indices"]
    reference_indices = batch["reference_indices"]
    target_indices = batch["target_indices"]
    if reservoir_indices.ndim != 2:
        raise ValueError("live reservoir indices must have shape [B, M]")
    batch_size, reservoir_count = reservoir_indices.shape
    if candidate_count < 2 or candidate_count >= reservoir_count:
        raise ValueError("live candidate count must be smaller than the reservoir")
    if not torch.equal(reservoir_indices[:, 0], target_indices):
        raise ValueError("live reservoir target must remain at position zero")

    flat_indices = torch.cat(
        [reference_indices, reservoir_indices.reshape(-1)], dim=0
    ).detach().cpu()
    unique_indices, inverse = torch.unique(
        flat_indices, sorted=True, return_inverse=True
    )
    unique_values = unique_indices.tolist()
    raw_feature_chunks: list[torch.Tensor] = []
    pixel_cache_wait_seconds = 0.0
    image_forward_started = time.perf_counter()
    cuda_forward_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    if pixel_cache is not None:
        if len(pixel_cache) != len(image_paths):
            raise ValueError("gallery pixel cache does not match the image split")
        pixel_iterator = iter(
            iter_prefetched_pixel_chunks(
                cache=pixel_cache,
                indices=unique_indices,
                batch_size=image_batch_size,
                prefetch_batches=transfer_prefetch_batches,
                pin_memory=pin_memory and device.type == "cuda",
            )
        )
        while True:
            wait_started = time.perf_counter()
            try:
                cached_pixels = next(pixel_iterator)
            except StopIteration:
                break
            pixel_cache_wait_seconds += time.perf_counter() - wait_started
            if device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            pixel_values = cached_pixels.to(
                device,
                non_blocking=cached_pixels.is_pinned(),
            )
            with _autocast(device, mixed_precision):
                raw_feature_chunks.append(
                    model._encode_image_raw(pixel_values).detach()
                )
            if device.type == "cuda":
                end_event.record()
                cuda_forward_events.append((start_event, end_event))
            del cached_pixels, pixel_values
    else:
        for begin in range(0, len(unique_values), image_batch_size):
            local_indices = unique_values[begin : begin + image_batch_size]
            images = _load_rgb_images(
                image_paths, local_indices, workers=image_loader_workers
            )
            pixel_values = processor(images=images, return_tensors="pt")[
                "pixel_values"
            ].to(device, non_blocking=True)
            del images
            with _autocast(device, mixed_precision):
                raw_feature_chunks.append(
                    model._encode_image_raw(pixel_values).detach()
                )
            del pixel_values
    if cuda_forward_events:
        torch.cuda.synchronize(device)
        image_forward_seconds = sum(
            start.elapsed_time(end) / 1000.0
            for start, end in cuda_forward_events
        )
    else:
        image_forward_seconds = time.perf_counter() - image_forward_started
    raw_features = torch.cat(raw_feature_chunks, dim=0)
    del raw_feature_chunks
    normalized_features = F.normalize(raw_features, dim=-1)
    reference_bank = (
        normalized_features
        if model.normalize_components_before_fusion
        else raw_features
    )
    reference_positions = inverse[:batch_size].to(device)
    candidate_positions = inverse[batch_size:].reshape(
        batch_size, reservoir_count
    ).to(device)
    with _autocast(device, mixed_precision):
        text_features = model.encode_text(
            batch["input_ids"], batch["attention_mask"]
        )
        queries = model.compose(
            reference_bank[reference_positions], text_features
        )
        candidate_features = normalized_features[candidate_positions]
        reservoir_scores = torch.einsum(
            "bd,bkd->bk", queries, candidate_features
        ).float()

    raw_topk_positions = torch.topk(
        reservoir_scores, k=candidate_count, dim=1
    ).indices
    raw_target_in_topk = (raw_topk_positions == 0).any(dim=1)
    target_scores = reservoir_scores[:, 0]
    live_target_ranks = 1 + (
        reservoir_scores[:, 1:] > target_scores[:, None]
    ).sum(dim=1)

    non_target_scores = reservoir_scores.clone()
    non_target_scores[:, 0] = -torch.inf
    selected_negative_positions = torch.topk(
        non_target_scores, k=candidate_count - 1, dim=1
    ).indices
    selected_positions = torch.cat(
        [
            torch.zeros(
                (batch_size, 1), dtype=torch.long, device=device
            ),
            selected_negative_positions,
        ],
        dim=1,
    )
    selected_candidate_indices = reservoir_indices.gather(1, selected_positions)
    selected_teacher_scores = None
    if "reservoir_teacher_scores" in batch:
        selected_teacher_scores = batch["reservoir_teacher_scores"].gather(
            1, selected_positions
        )
    selected_anchor_scores = None
    if "reservoir_anchor_scores" in batch:
        selected_anchor_scores = batch["reservoir_anchor_scores"].gather(
            1, selected_positions
        )

    metrics = {
        "live_reservoir_candidate_count": reservoir_scores.new_tensor(
            float(reservoir_count)
        ),
        "live_reservoir_unique_images_per_batch": reservoir_scores.new_tensor(
            float(len(unique_values))
        ),
        "live_reservoir_raw_target_recall_at_k": (
            raw_target_in_topk.float().mean()
        ),
        "live_reservoir_target_mean_rank": live_target_ranks.float().mean(),
        "live_reservoir_target_recall_at_reservoir_k": (
            batch["raw_target_in_reservoir"].float().mean()
        ),
        "live_reservoir_pixel_cache_wait_seconds": reservoir_scores.new_tensor(
            pixel_cache_wait_seconds
        ),
        "live_reservoir_image_forward_seconds": reservoir_scores.new_tensor(
            image_forward_seconds
        ),
    }
    return {
        "selected_candidate_indices": selected_candidate_indices.detach().cpu(),
        "selected_teacher_scores": (
            None if selected_teacher_scores is None
            else selected_teacher_scores.detach().cpu()
        ),
        "selected_anchor_scores": (
            None
            if selected_anchor_scores is None
            else selected_anchor_scores.detach().cpu()
        ),
        "raw_target_in_topk": raw_target_in_topk.detach(),
        "metrics": metrics,
    }


def _materialize_selected_candidate_batch(
    *,
    processor,
    source_batch: dict[str, Any],
    selected_candidate_indices: torch.Tensor,
    selected_anchor_scores: torch.Tensor | None,
    raw_target_in_topk: torch.Tensor,
    image_paths: list[Path],
    preprocess_batch_size: int,
    image_loader_workers: int,
    device: torch.device,
    pixel_cache: GalleryPixelCache | None = None,
    transfer_prefetch_batches: int = 2,
    pin_memory: bool = True,
    selected_teacher_scores: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Decode only the selected Top-K again so its forward pass keeps gradients."""

    reference_indices = source_batch["reference_indices"].detach().cpu()
    candidates = selected_candidate_indices.detach().cpu()
    batch_size, candidate_count = candidates.shape
    all_indices = torch.cat(
        [reference_indices, candidates.reshape(-1)], dim=0
    )
    unique_indices, inverse = torch.unique(
        all_indices, sorted=True, return_inverse=True
    )
    unique_values = unique_indices.tolist()
    pixel_chunks: list[torch.Tensor] = []
    if pixel_cache is not None:
        if len(pixel_cache) != len(image_paths):
            raise ValueError("gallery pixel cache does not match the image split")
        for cached_pixels in iter_prefetched_pixel_chunks(
            cache=pixel_cache,
            indices=unique_indices,
            batch_size=preprocess_batch_size,
            prefetch_batches=transfer_prefetch_batches,
            pin_memory=pin_memory and device.type == "cuda",
        ):
            pixel_chunks.append(
                cached_pixels.to(
                    device,
                    non_blocking=cached_pixels.is_pinned(),
                )
            )
            del cached_pixels
    else:
        for begin in range(0, len(unique_values), preprocess_batch_size):
            local_indices = unique_values[begin : begin + preprocess_batch_size]
            images = _load_rgb_images(
                image_paths, local_indices, workers=image_loader_workers
            )
            pixels = processor(images=images, return_tensors="pt")["pixel_values"]
            pixel_chunks.append(pixels.to(device, non_blocking=True))
            del images, pixels
    unique_pixel_values = torch.cat(pixel_chunks, dim=0)
    del pixel_chunks
    reference_positions = inverse[:batch_size].to(device)
    candidate_positions = inverse[batch_size:].reshape(
        batch_size, candidate_count
    ).to(device)
    result: dict[str, Any] = {
        "pair_ids": source_batch["pair_ids"],
        "unique_pixel_values": unique_pixel_values,
        "reference_image_positions": reference_positions,
        "candidate_image_positions": candidate_positions,
        "candidate_is_negative_query": torch.zeros(
            (batch_size, candidate_count), dtype=torch.bool, device=device
        ),
        "unique_image_count": torch.tensor(
            len(unique_values), dtype=torch.long, device=device
        ),
        "input_ids": source_batch["input_ids"],
        "attention_mask": source_batch["attention_mask"],
        "negative_query_input_ids": None,
        "negative_query_attention_mask": None,
        "negative_query_owners": torch.empty(
            (0,), dtype=torch.long, device=device
        ),
        "negative_query_positions": torch.empty(
            (0,), dtype=torch.long, device=device
        ),
        "raw_target_in_topk": raw_target_in_topk.to(device),
    }
    if selected_anchor_scores is not None:
        result["anchor_scores"] = selected_anchor_scores.to(
            device, non_blocking=True
        )
    if selected_teacher_scores is not None:
        if selected_teacher_scores.shape != candidates.shape:
            raise ValueError("selected Teacher scores must match live candidates")
        result["teacher_scores"] = selected_teacher_scores.detach().to(
            device, non_blocking=True
        )
        result["teacher_target_top1"] = result["teacher_scores"].argmax(dim=1) == 0
    return result


def _load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _load_rgb_images(
    image_paths: list[Path],
    indices: list[int],
    *,
    workers: int,
) -> list[Image.Image]:
    paths = [image_paths[index] for index in indices]
    if workers <= 1:
        return [_load_rgb_image(path) for path in paths]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_load_rgb_image, paths))


def _compute_live_student_scores(
    *,
    model: ComposedRetrievalModel,
    batch: dict[str, Any],
    live_image_batch_size: int,
) -> torch.Tensor:
    """Re-encode the selected minibatch candidates with the current Student."""

    raw_image_features = torch.cat(
        [
            model._encode_image_raw(pixel_chunk)
            for pixel_chunk in batch["unique_pixel_values"].split(
                live_image_batch_size
            )
        ],
        dim=0,
    )
    normalized_image_features = F.normalize(raw_image_features, dim=-1)
    reference_bank = (
        normalized_image_features
        if model.normalize_components_before_fusion
        else raw_image_features
    )
    reference_features = reference_bank[batch["reference_image_positions"]]
    candidates = model.target_from_raw(raw_image_features)[batch["candidate_image_positions"]]
    text_features = model.encode_text(batch["input_ids"], batch["attention_mask"])
    queries = model.compose(reference_features, text_features)
    negative_query_input_ids = batch["negative_query_input_ids"]
    if negative_query_input_ids is not None and negative_query_input_ids.shape[0] > 0:
        negative_text_features = model.encode_text(
            negative_query_input_ids,
            batch["negative_query_attention_mask"],
        )
        negative_query_features = model.compose(
            reference_features[batch["negative_query_owners"]],
            negative_text_features,
        )
    else:
        negative_query_features = queries.new_empty((0, queries.shape[-1]))
    candidates = merge_mixed_candidate_features(
        image_candidate_features=candidates,
        negative_query_features=negative_query_features,
        negative_query_owners=batch["negative_query_owners"],
        negative_query_positions=batch["negative_query_positions"],
        candidate_is_negative_query=batch["candidate_is_negative_query"],
    )
    return torch.einsum("bd,bkd->bk", queries, candidates)


def _disable_dropout_randomness(model: nn.Module) -> int:
    """Disable stochastic policy noise without changing trainability or other modes."""

    count = 0
    for module in model.modules():
        if isinstance(module, nn.modules.dropout._DropoutNd):
            module.eval()
            count += 1
    return count


def _write_training_log(
    *,
    output_dir: Path,
    epoch_number: int,
    global_step: int,
    scheduler: LambdaLR,
    mining_metadata: dict[str, Any],
    config: OnPolicyDistillationConfig,
    running: dict[str, float],
    running_count: int,
) -> None:
    learning_rates = scheduler.get_last_lr()
    payload: dict[str, Any] = {
        "epoch_number": epoch_number,
        "global_step": global_step,
        "head_learning_rate": learning_rates[0],
        "backbone_learning_rate": (
            learning_rates[1] if len(learning_rates) > 1 else None
        ),
        "mining_raw_target_recall_at_k": mining_metadata[
            "raw_target_recall_at_k"
        ],
        "candidate_source": config.mining.candidate_source,
        "candidate_refresh_mode": config.mining.refresh_mode,
        "mining_teacher_target_top1_rate": mining_metadata.get(
            "teacher_target_top1_rate"
        ),
        "mining_teacher_target_top10_rate": mining_metadata.get(
            "teacher_target_top10_rate"
        ),
        "mining_teacher_target_mean_rank": mining_metadata.get(
            "teacher_target_mean_rank"
        ),
        "mining_teacher_rank_entropy": mining_metadata.get(
            "teacher_rank_entropy"
        ),
        "mining_teacher_score_mean": mining_metadata.get("teacher_score_mean"),
        "mining_teacher_score_std": mining_metadata.get("teacher_score_std"),
    }
    payload.update(
        {
            key: round(value / running_count, 7)
            for key, value in running.items()
        }
    )
    serialized = json.dumps(payload, ensure_ascii=False)
    print(serialized, flush=True)
    with (output_dir / "train_metrics.jsonl").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(serialized + "\n")


def _audit_trainable_gradients(model: ComposedRetrievalModel) -> dict[str, int]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return {
        "trainable_tensor_count": len(parameters),
        "trainable_parameter_count": sum(parameter.numel() for parameter in parameters),
        "nonzero_gradient_tensor_count": sum(
            parameter.grad is not None and bool((parameter.grad != 0).any())
            for parameter in parameters
        ),
        "missing_gradient_count": sum(parameter.grad is None for parameter in parameters),
        "nonfinite_gradient_count": sum(
            parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
            for parameter in parameters
        ),
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def _autocast(device: torch.device, mixed_precision: str):
    if device.type != "cuda" or mixed_precision == "no":
        return nullcontext()
    dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)
