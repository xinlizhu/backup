"""LLaVA shared-LoRA training with disjoint queries and synchronized gradients."""
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from cir_qdiag_train import qwen_on_policy_trainer as engine
from cir_qdiag_train.llava_parallel import (
    average_metrics, broadcast_parameters, on_rank_zero, setup_parallel, shard_order, sum_gradients,
)
from cir_qdiag_train.llava_student import LlavaLoRAStudent
from cir_qdiag_train.on_policy_mining import write_json_atomic
from cir_qdiag_train.qwen_on_policy_config import QwenOnPolicyDistillationConfig
from cir_qdiag_train.qwen_student import load_qwen_checkpoint, save_qwen_checkpoint
from cir_qdiag_train.resume_config import without_loader_settings


def _validate_output(config, resume_from):
    output = config.experiment.output_dir
    path = output / "effective_config.json"
    if path.exists():
        old = QwenOnPolicyDistillationConfig.model_validate(json.loads(path.read_text()))
        if without_loader_settings(old.model_dump(mode="json")) != without_loader_settings(config.model_dump(mode="json")):
            raise ValueError(f"Refusing changed configuration in {output}")
        if resume_from is None and (output / "last.pt").exists():
            raise ValueError("Existing checkpoint: use --resume-from or a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, config.model_dump(mode="json"))


def train_llava_parallel(config, *, resume_from=None, smoke_only=False, allow_cpu=False):
    if not (config.model.llava or config.model.variant in {"qwen3_vl_embedding_shared_lora", "qwen3_vl_shared_lora"}) or config.optimizer.gradient_accumulation_steps != 1:
        raise ValueError("Requires shared LLaVA or Qwen LoRA with accumulation=1")
    if config.objective.uses_teacher_filter:
        raise ValueError("Parallel LLaVA uses unfiltered queries")
    if smoke_only and resume_from is not None:
        raise ValueError("Smoke does not resume or write training checkpoints")
    rank, world, device = setup_parallel(config.optimizer.world_size, allow_cpu=allow_cpu)
    temporary = None
    try:
        # The smoke uses one global batch of random InfoNCE, just like the old
        # smoke, never running full-gallery mining or mutating formal outputs.
        if smoke_only:
            temporary = on_rank_zero(lambda: tempfile.mkdtemp(prefix="llava_parallel_smoke_"))
            config = config.model_copy(deep=True)
            config.experiment.output_dir = Path(temporary)
            config.optimizer.epochs = config.optimizer.save_every_epochs = 1
            config.objective.loss_type = "candidate_infonce"
            config.objective.rank_temperature = .07
            config.mining.candidate_source = "random"
        return _train(config, resume_from=resume_from, smoke_only=smoke_only,
                      rank=rank, world=world, device=device)
    finally:
        # This exact, newly allocated scratch directory belongs to this smoke.
        # It never contains a training checkpoint or an existing run's files.
        if temporary is not None and rank == 0:
            shutil.rmtree(temporary)
        dist.destroy_process_group()


def _train(config, *, resume_from, smoke_only, rank, world, device):
    on_rank_zero(lambda: _validate_output(config, resume_from))
    engine._set_seed(config.experiment.seed)
    output = config.experiment.output_dir
    image_ids, paths = engine.load_image_map(config.data.train_image_split_path, config.data.image_root)
    records = engine.load_cirr_positive_records(config.data.train_annotation_path, image_ids, paths)
    local_batch = config.optimizer.batch_size_per_gpu
    global_batch = local_batch * world
    if smoke_only:
        records = records[:global_batch]
        if len(records) != global_batch:
            raise ValueError("Not enough queries for a complete global smoke batch")
    if not records:
        raise ValueError("No training queries")
    from cir_qdiag_train.qwen_student import QwenQueryLoRAStudent
    student_class = LlavaLoRAStudent if config.model.llava else QwenQueryLoRAStudent
    student = student_class(config.model, device=device, mixed_precision=config.optimizer.mixed_precision)
    parameters = student.trainable_parameters()
    from cir_qdiag_train.llava_parallel import OverlappedGradientSum
    overlap = (OverlappedGradientSum(parameters)
               if config.model.variant in {"qwen3_vl_embedding_shared_lora", "qwen3_vl_shared_lora"}
               and not config.optimizer.candidate_gradient_cache
               and os.environ.get("QWEN_OVERLAP_GRADIENTS", "1") != "0" else None)
    audit = student.trainability_audit() if config.model.llava else engine.audit_shared_adapters(student)
    model_label = "LLaVA" if config.model.llava else ("Qwen Embedding" if config.model.variant == "qwen3_vl_embedding_shared_lora" else "Qwen Instruct")
    from cir_qdiag_train.step_profile import StepProfile, prepared_stats
    profile_steps = max(0, int(os.environ.get("QWEN_PROFILE_STEPS", "0")))
    profile_start = max(1, int(os.environ.get("QWEN_PROFILE_START", "5")))
    process_step = 0
    profile_path = output / f"step_profile_rank{rank}.jsonl"
    if profile_steps and rank == 0:
        print(f"[profile] process steps {profile_start}..{profile_start + profile_steps - 1}; "
              f"per-rank files: {output}/step_profile_rank*.jsonl", flush=True)
    on_rank_zero(lambda: write_json_atomic(output / "adapter_audit.json", audit))
    optimizer = torch.optim.AdamW(parameters, lr=config.optimizer.learning_rate,
                                  weight_decay=config.optimizer.weight_decay)
    steps_per_epoch = math.ceil(len(records) / global_batch)
    total_steps = steps_per_epoch * config.optimizer.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, engine._cosine_schedule(
        total_steps=total_steps, warmup_steps=int(total_steps * config.optimizer.warmup_ratio)))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and config.optimizer.mixed_precision == "fp16")
    start_epoch, global_step = 0, 0
    if resume_from is not None:
        checkpoint = load_qwen_checkpoint(resume_from, student=student, optimizer=optimizer,
                                         scheduler=scheduler, scaler=scaler,
                                         expected_config=config.model_dump(mode="json"))
        start_epoch, global_step = int(checkpoint["epoch"]) + 1, int(checkpoint["global_step"])
        del checkpoint
    broadcast_parameters(parameters)
    on_rank_zero(lambda: write_json_atomic(output / "run_summary.json", {
        "algorithm": config.objective.loss_type, "rank_kl_direction": config.objective.rank_kl_direction,
        "student_initialization": config.model.pretrained_model_name_or_path,
        "student_architecture": config.model.variant, "parallel_mode": "synchronous LoRA gradient all-reduce",
        "world_size": world, "batch_size_per_gpu": local_batch, "gradient_accumulation_steps": 1,
        "gradient_communication_overlap": overlap is not None,
        "gradient_bucket_mib": 8 if overlap is not None else 32,
        "effective_batch_size": global_batch, "optimizer_steps_per_epoch": steps_per_epoch,
        "total_optimizer_steps": total_steps, "candidate_count_per_query": config.mining.top_k,
        "candidate_source": config.mining.candidate_source, "target_forced_into_candidate_set": True,
        "teacher_target_max_rank": None, "candidate_gradient_cache": config.optimizer.candidate_gradient_cache,
        "gallery_policy": "rank 0 refreshes current gallery each epoch; both ranks backpropagate live candidates",
        "data_loader_workers_per_rank": config.data.num_workers, "prefetch_factor": config.data.prefetch_factor,
        "sampler": "disjoint global-batch shards; no padding or dropping; partial batches query-weighted",
        "epoch_rng": "seed + epoch_index + rank * 1000003; reset after mining, also on epoch resume",
        **student.parameter_summary(),
    }))
    if rank == 0:
        print(f"[gradient sync] overlap={overlap is not None}; prefetch={config.data.prefetch_factor}; "
              f"workers_per_rank={config.data.num_workers}", flush=True)
        print(f"{model_label} synchronized training: {world} GPUs x {local_batch} queries; "
              f"global_batch={global_batch}; accumulation=1; K={config.mining.top_k}", flush=True)
    timings = []
    for epoch_index in range(start_epoch, config.optimizer.epochs):
        epoch_started = time.perf_counter()
        epoch = epoch_index + 1

        def mine():
            directory, metadata = engine.refresh_qwen_candidates(
                config=config, student=student, records=records, image_ids=image_ids,
                gallery=None, image_paths=paths, epoch=epoch, global_step=global_step, device=device)
            return str(directory), metadata

        directory, mining_metadata = on_rank_zero(mine)
        mining_seconds = time.perf_counter() - epoch_started
        directory = Path(directory)
        arrays = {name: np.load(directory / f"{name}.npy", mmap_mode="r") for name in
                  ("candidate_indices", "raw_target_in_topk", "teacher_scores", "teacher_target_top1")}
        engine._set_seed(config.experiment.seed + epoch_index + rank * 1000003)
        student.train()
        order = torch.randperm(len(records), generator=torch.Generator().manual_seed(
            config.experiment.seed + epoch_index)).tolist()
        local_order = shard_order(order, local_batch, rank, world)
        batches = iter(engine.shared_training_batches(
            student=student, records=records, order=local_order, candidates=arrays["candidate_indices"],
            image_paths=paths, batch_size=local_batch,
            image_batch_size=config.optimizer.candidate_encode_batch_size,
            workers=config.data.num_workers, prefetch_factor=config.data.prefetch_factor,
            seed=config.experiment.seed + epoch_index + rank * 1000003))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            if smoke_only:
                torch.cuda.reset_peak_memory_stats(device)
        training_started = time.perf_counter()
        wait_seconds, compute_seconds = 0., 0.
        running, running_queries = {}, 0
        for step in tqdm(range(steps_per_epoch), desc=f"{model_label} {world}GPU train epoch {epoch}", disable=rank != 0):
            optimizer.zero_grad(set_to_none=True)
            actual_global = min(global_batch, len(records) - step * global_batch)
            local_count = max(0, min(local_batch, actual_global - rank * local_batch))
            process_step += 1
            profile = (StepProfile(device)
                       if profile_start <= process_step < profile_start + profile_steps else None)
            waited = time.perf_counter()
            indices, prepared = next(batches) if local_count else ([], None)
            wait_seconds += time.perf_counter() - waited
            if len(indices) != local_count:
                raise RuntimeError("Data-parallel shard lost alignment")
            compute_started = time.perf_counter()
            stats = prepared_stats(prepared) if profile else {}
            if profile:
                profile.compute_start()
            metrics = {}
            if overlap is not None:
                overlap.begin()
            if local_count:
                if config.optimizer.candidate_gradient_cache:
                    scores, replay = engine.shared_student_scores(student, prepared, gradient_cache=True)
                else:
                    scores, replay = engine.compute_live_qwen_scores(student, prepared), None
                def tensor(name, dtype):
                    return torch.tensor(np.array(arrays[name][indices], copy=True), device=device, dtype=dtype)
                loss, raw_metrics = engine.qwen_candidate_loss(
                    config=config, student_scores=scores, teacher_scores=tensor("teacher_scores", torch.float32),
                    teacher_target_top1=tensor("teacher_target_top1", torch.bool),
                    raw_target_in_topk=tensor("raw_target_in_topk", torch.bool),
                    rank_weight=config.objective.rank_weight, centered_response_weight=0.,
                    rank_temperature=config.objective.rank_temperature,
                    centered_response_beta=config.objective.centered_response_beta,
                    require_teacher_target_top1=False, teacher_target_max_rank=None,
                    rank_kl_direction=config.objective.rank_kl_direction)
                if profile:
                    profile.mark("forward_loss")
                scaler.scale(loss * (local_count / actual_global)).backward()
                if replay is not None:
                    replay()
                if profile:
                    profile.mark("backward_replay")
                del loss, scores, replay, prepared
            elif profile:
                profile.mark("forward_loss")
                profile.mark("backward_replay")
            launched = len(overlap.pending) if overlap is not None else 0
            if overlap is not None:
                if step == 0 and rank == 0:
                    print(f"[gradient sync] buckets launched during backward: "
                          f"{len(overlap.pending)}/{len(overlap.buckets)}", flush=True)
                overlap.finish()
            else:
                sum_gradients(parameters)
            if profile:
                profile.mark("gradient_finish")
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, config.optimizer.max_grad_norm, error_if_nonfinite=True)
            if step == 0:
                gradient_audit = engine._audit_gradients(student)
                inactive = audit.get("inactive_vision_adapter_modules", [])
                missing = [n for n, p in student.model.named_parameters() if p.requires_grad and p.grad is None
                           and not any(n.startswith(prefix + ".") for prefix in inactive)]
                if missing or gradient_audit["nonfinite_gradient_count"]:
                    raise RuntimeError(f"Invalid {model_label} gradients: {missing}, {gradient_audit}")
                if smoke_only:
                    for component in (("language_model", "vision_tower", "multi_modal_projector") if config.model.llava
                                      else ("language_model", "visual.blocks", "merger")):
                        if not any(p.grad is not None and bool(p.grad.count_nonzero())
                                   for n, p in student.model.named_parameters() if p.requires_grad and component in n):
                            raise RuntimeError(f"Smoke found no nonzero gradient in {component}")
                on_rank_zero(lambda: write_json_atomic(output / "gradient_audit.json", gradient_audit))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if profile:
                profile.mark("optimizer")
            if local_count:
                metrics = {k: float(v.item()) for k, v in raw_metrics.items()}
                del raw_metrics
            if profile:
                profile.mark("metrics")
                measured = profile.finish()
                row_profile = {"rank": rank, "pid": os.getpid(), "epoch": epoch,
                               "global_step": global_step + 1, "process_step": process_step,
                               "local_queries": local_count, "overlap": overlap is not None,
                               "buckets_launched_during_backward": launched, **stats,
                               **{k: round(v, 6) for k, v in measured.items()}}
                with profile_path.open("a") as handle:
                    handle.write(json.dumps(row_profile) + "\n")
                print("[step profile] " + json.dumps(row_profile), flush=True)
            global_step += 1
            if device.type == "cuda" and smoke_only:
                torch.cuda.synchronize(device)
            compute_seconds += time.perf_counter() - compute_started
            for key, value in metrics.items():
                running[key] = running.get(key, 0.) + value * local_count
            running_queries += local_count
            if global_step % config.optimizer.log_every_steps == 0 or step == steps_per_epoch - 1:
                mean = average_metrics({k: v / max(running_queries, 1) for k, v in running.items()},
                                       running_queries, device)
                if rank == 0:
                    row = {"epoch_number": epoch, "global_step": global_step, "global_batch_size": global_batch,
                           "learning_rate": scheduler.get_last_lr()[0],
                           "mining_raw_target_recall_at_k": mining_metadata["raw_target_recall_at_k"],
                           **{k: round(v, 7) for k, v in mean.items()}}
                    print(json.dumps(row), flush=True)
                    with (output / "train_metrics.jsonl").open("a") as handle:
                        handle.write(json.dumps(row) + "\n")
                running, running_queries = {}, 0
        batches.close()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        training_seconds = time.perf_counter() - training_started
        if smoke_only:
            local_report = {"rank": rank, "device": str(device), "queries": len(local_order),
                            "seconds_one_update": training_seconds, "data_wait_seconds": wait_seconds,
                            "compute_and_sync_seconds": compute_seconds, "gradient_audit": gradient_audit,
                            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None,
                            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else None}
            reports = [None] * world
            dist.all_gather_object(reports, local_report)
            if rank == 0:
                print(json.dumps({"smoke_only": True, "saved_checkpoint": False, "objective": "candidate_infonce",
                                  "global_batch_size": global_batch, "batch_size_per_gpu": local_batch,
                                  "world_size": world, "gradient_accumulation_steps": 1,
                                  "candidate_gradient_cache": config.optimizer.candidate_gradient_cache,
                                  "num_workers_per_rank": config.data.num_workers, "ranks": reports}, indent=2), flush=True)
            if overlap is not None:
                overlap.close()
            return None
        timings.append({"epoch": epoch, "mining_seconds": mining_seconds, "training_seconds": training_seconds,
                        "total_seconds": time.perf_counter() - epoch_started})

        def save():
            write_json_atomic(output / "training_time.json", {"epochs_this_process": timings, "completed_epochs": epoch})
            if epoch % config.optimizer.save_every_epochs == 0 or epoch == config.optimizer.epochs:
                kwargs = dict(student=student, epoch=epoch_index, global_step=global_step,
                              config=config.model_dump(mode="json"), scaler=scaler)
                save_qwen_checkpoint(output / f"checkpoint_epoch_{epoch:03d}.pt", optimizer=None, scheduler=None, **kwargs)
                save_qwen_checkpoint(output / "last.pt", optimizer=optimizer, scheduler=scheduler, **kwargs)
        on_rank_zero(save)
    if overlap is not None:
        overlap.close()
    return output / "last.pt"
