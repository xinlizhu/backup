"""One accumulated optimizer update, including data/compute timing; saves no run."""
# ruff: noqa: E402 -- bootstrap the local package before importing training code

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

import numpy as np
import torch
import torch.nn.functional as F

from cir_qdiag_train.on_policy_data import load_cirr_positive_records, load_image_map
from cir_qdiag_train.on_policy_mining import _sample_random_candidates
from cir_qdiag_train.qwen_gradient_cache import shared_student_scores
from cir_qdiag_train.qwen_live_training import compute_live_qwen_scores
from cir_qdiag_train.qwen_on_policy_config import load_qwen_on_policy_config
from cir_qdiag_train.qwen_on_policy_trainer import _audit_gradients, _set_seed
from cir_qdiag_train.qwen35_student import Qwen35Student
from cir_qdiag_train.qwen_training_data import shared_training_batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["lora", "full"], default="lora")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, help="default: config data.num_workers")
    parser.add_argument("--micro-steps", type=int, help="default: the configured gradient accumulation steps")
    parser.add_argument("--batch-size", type=int, help="override query micro-batch for this smoke test only")
    parser.add_argument("--candidate-batch-size", type=int, help="override image chunk size for this smoke test only")
    args = parser.parse_args()
    if args.num_workers is not None and args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")
    config = load_qwen_on_policy_config(args.config or PROJECT / f"configs/qwen35_{args.mode}_oprd_s2.yaml")
    if args.num_workers is None:
        args.num_workers = config.data.num_workers
    if not config.model.qwen35:
        parser.error("requires a Qwen3.5 config")
    for value in [args.micro_steps, args.batch_size, args.candidate_batch_size]:
        if value is not None and value < 1:
            parser.error("micro-steps and batch sizes must be positive")
    if args.batch_size is not None:
        config.optimizer.batch_size_per_gpu = args.batch_size
    if args.candidate_batch_size is not None:
        config.optimizer.candidate_encode_batch_size = args.candidate_batch_size
    micro_steps = args.micro_steps or config.optimizer.gradient_accumulation_steps
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run this smoke test on the training GPU")
    _set_seed(config.experiment.seed)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    image_ids, paths = load_image_map(config.data.train_image_split_path, config.data.image_root)
    records = load_cirr_positive_records(config.data.train_annotation_path, image_ids, paths)
    records = records[:config.optimizer.batch_size_per_gpu * micro_steps]
    if len(records) != config.optimizer.batch_size_per_gpu * micro_steps:
        raise ValueError("Not enough records for the requested smoke update")
    index = {name: i for i, name in enumerate(image_ids)}
    candidates, _ = _sample_random_candidates(
        sample_count=len(records), gallery_count=len(paths),
        reference_indices=np.array([index[r.reference_id] for r in records]),
        target_indices=np.array([index[r.target_id] for r in records]),
        top_k=config.mining.top_k, seed=config.experiment.seed,
    )
    student = Qwen35Student(config.model, device=device,
                                   mixed_precision=config.optimizer.mixed_precision)
    audit = student.trainability_audit()
    optimizer = torch.optim.AdamW(student.trainable_parameters(), lr=config.optimizer.learning_rate,
                                  weight_decay=config.optimizer.weight_decay)
    batches = iter(shared_training_batches(
        student=student, records=records, order=list(range(len(records))), candidates=candidates,
        image_paths=paths, batch_size=config.optimizer.batch_size_per_gpu,
        image_batch_size=config.optimizer.candidate_encode_batch_size,
        workers=args.num_workers, prefetch_factor=config.data.prefetch_factor,
        seed=config.experiment.seed,
    ))
    student.train()
    optimizer.zero_grad(set_to_none=True)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and config.optimizer.mixed_precision == "fp16")
    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    synchronize()
    started = time.perf_counter()
    data_seconds = 0.0
    compute_seconds = 0.0
    losses = []
    for _ in range(micro_steps):
        phase_started = time.perf_counter()
        _, prepared = next(batches)
        data_seconds += time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        if config.optimizer.candidate_gradient_cache:
            scores, replay = shared_student_scores(student, prepared, gradient_cache=True)
        else:
            scores = compute_live_qwen_scores(student, prepared)
            replay = None
        # A labelled loss exercises both branches; this is not an OPRD evaluation.
        loss = F.cross_entropy(scores / .07, torch.zeros(scores.shape[0], dtype=torch.long, device=device))
        scaler.scale(loss / micro_steps).backward()
        if replay is not None:
            replay()
        synchronize()
        compute_seconds += time.perf_counter() - phase_started
        losses.append(loss.item())
        del scores, loss, prepared, replay
    optimizer_started = time.perf_counter()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(student.trainable_parameters(), config.optimizer.max_grad_norm,
                                   error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    synchronize()
    optimizer_seconds = time.perf_counter() - optimizer_started
    update_seconds = time.perf_counter() - started
    gradient_audit = _audit_gradients(student)
    if gradient_audit["nonfinite_gradient_count"] or not gradient_audit["nonzero_gradient_tensor_count"]:
        raise RuntimeError(f"Invalid gradients: {gradient_audit}")
    for label, fragment in [("vision", "visual.blocks"), ("merger", "merger"), ("language", "language_model")]:
        if not any(p.grad is not None and bool(p.grad.abs().sum() > 0)
                   for n, p in student.model.named_parameters() if fragment in n and p.requires_grad):
            raise RuntimeError(f"No nonzero gradient in {label}")
    print(json.dumps({
        "smoke_only": True, "objective": "candidate_infonce", "saved_checkpoint": False,
        "loss": sum(losses) / len(losses), "seconds_one_update": update_seconds,
        "data_wait_seconds": data_seconds, "forward_backward_seconds": compute_seconds,
        "optimizer_step_seconds": optimizer_seconds,
        "query_micro_batch": config.optimizer.batch_size_per_gpu,
        "micro_steps_in_update": micro_steps, "queries_in_update": len(records),
        "matches_configured_accumulation": micro_steps == config.optimizer.gradient_accumulation_steps,
        "candidate_encode_batch_size": config.optimizer.candidate_encode_batch_size,
        "candidate_gradient_cache": config.optimizer.candidate_gradient_cache,
        "num_workers": args.num_workers,
        "seconds_per_query": update_seconds / len(records),
        "mode": audit["mode"], "trainable_parameters": audit["trainable_parameter_count"],
        "linear_modules_by_component": {k: len(v) for k, v in audit["modules"].items()},
        "fla_delta_kernel_available": audit["fla_delta_kernel_available"],
        "causal_conv1d_kernel_available": audit["causal_conv1d_kernel_available"],
        "gradient_audit": gradient_audit,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else None,
    }, indent=2), flush=True)
    batches.close()


if __name__ == "__main__":
    main()
