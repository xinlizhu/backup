#!/usr/bin/env python3
"""Train TG-CIR from public CLIP initialization, with automatic step resume."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

import torch
from tqdm import tqdm

from cir_qdiag_train.on_policy_mining import write_json_atomic
from cir_qdiag_train.tg_cir_config import (
    DEFAULT_CONFIG,
    load_config,
    model_label,
    training_identity,
)
from cir_qdiag_train.tg_cir_data import TGCIRProcessor, build_train_loader
from cir_qdiag_train.tg_cir_evaluation import cirr_split, evaluate_cirr
from cir_qdiag_train.tg_cir_model import total_loss
from cir_qdiag_train.tg_cir_runtime import (
    build_model,
    evaluation_identity,
    link_checkpoint,
    load_checkpoint,
    output_lock,
    restore_rng,
    rng_state,
    save_checkpoint,
    seed_all,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate inputs; no model/training/output"
    )
    parser.add_argument(
        "--smoke-test", action="store_true", help="Two batches only; no checkpoints/output"
    )
    return parser.parse_args()


def best_validation_score(output):
    best = -float("inf")
    for path in output.glob("val_epoch_*.json"):
        values = json.loads(path.read_text())
        best = max(best, sum(v for k, v in values.items() if "recall_at_" in k))
    return best


def run(config, records, args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU visible. Use --dry-run here; train on your GPU host.")
    torch.set_num_threads(2)
    seed_all(config["seed"])
    options = config["training"]
    output = Path(config["output_dir"])
    last = output / "last.pt"
    previous = None if args.smoke_test or not last.is_file() else load_checkpoint(last, config)
    model = build_model(config, None if previous is None else previous["clip_config"])
    if previous is not None:
        model.load_state_dict(previous["model"], strict=True)
    model.to(device)
    backbone = list(model.backbone.clip.parameters())
    backbone_ids = {id(p) for p in backbone}
    heads = [p for p in model.parameters() if id(p) not in backbone_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone, "lr": options["clip_learning_rate"]},
            {"params": heads, "lr": options["learning_rate"]},
        ],
        weight_decay=options["weight_decay"],
        eps=1e-8,
    )
    use_amp = device.type == "cuda" and options["mixed_precision"] != "no"
    amp_dtype = torch.bfloat16 if options["mixed_precision"] == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    epoch, next_step, global_step, loss_sum = 1, 0, 0, 0.0
    if previous is not None:
        optimizer.load_state_dict(previous["optimizer"])
        scaler.load_state_dict(previous["scaler"])
        epoch, next_step = previous["epoch"], previous["next_step"]
        global_step, loss_sum = previous["global_step"], previous["loss_sum"]
        restore_rng(previous["rng"])
        print(
            f"[resume] {last}: epoch={epoch}, completed_batches={next_step}, "
            f"global_step={global_step}",
            flush=True,
        )
        del previous
    processor = TGCIRProcessor(config["model"]["pretrained_model_name_or_path"])
    full_steps = len(records) // options["batch_size"]
    stop = {"signal": None}

    def request_stop(signum, frame):
        del frame
        stop["signal"] = signum
        print("[stop requested] Saving after the current optimizer step.", flush=True)

    def save(epoch_number, completed_steps, running_loss):
        save_checkpoint(
            last,
            {
                "method": config["method"],
                "config": config,
                "clip_config": model.backbone.clip.config.to_dict(),
                "epoch": epoch_number,
                "next_step": completed_steps,
                "global_step": global_step,
                "loss_sum": running_loss,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "rng": rng_state(),
            },
        )

    old_handlers = {}
    if not args.smoke_test:
        old_handlers = {
            sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)
        }
    try:
        best = best_validation_score(output) if not args.smoke_test else -float("inf")
        for epoch_number in range(epoch, options["epochs"] + 1):
            # Released code decays before epochs 5 and 10; neither occurs in E4.
            factor = 0.1 ** (int(epoch_number >= 5) + int(epoch_number >= 10))
            optimizer.param_groups[0]["lr"] = options["clip_learning_rate"] * factor
            optimizer.param_groups[1]["lr"] = options["learning_rate"] * factor
            if next_step < full_steps:
                loader = build_train_loader(
                    records,
                    processor,
                    config,
                    epoch_number,
                    next_step,
                    device.type == "cuda",
                )
                print(
                    f"[train] E{epoch_number}/{options['epochs']} batch={options['batch_size']} "
                    f"workers={options['num_workers']} prefetch={options['prefetch_factor']} "
                    f"steps={full_steps} start={next_step}",
                    flush=True,
                )
                model.train()
                started = time.perf_counter()
                initial_step = next_step
                progress = tqdm(
                    loader, total=full_steps, initial=next_step, desc=f"TG-CIR E{epoch_number}"
                )
                for step, batch in enumerate(progress, start=next_step + 1):
                    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                        losses = model(**batch)
                        loss = total_loss(losses, config["loss_weights"])
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"Non-finite TG-CIR loss at E{epoch_number} step {step}"
                        )
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    global_step += 1
                    loss_sum += float(loss.detach())
                    progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
                    if (
                        step % options["log_every_steps"] == 0
                        or step == full_steps
                        or args.smoke_test
                    ):
                        report = {
                            "epoch": epoch_number,
                            "step": step,
                            "global_step": global_step,
                            "loss": float(loss.detach()),
                            "seconds_per_step": (time.perf_counter() - started)
                            / (step - initial_step),
                            **{name: float(value.detach()) for name, value in losses.items()},
                        }
                        print(json.dumps(report), flush=True)
                        if not args.smoke_test:
                            with (output / "train_steps.jsonl").open("a") as handle:
                                handle.write(json.dumps(report) + "\n")
                    if args.smoke_test:
                        if step - initial_step >= 2:
                            print(
                                "[smoke test passed] Two optimizer steps; no checkpoint written.",
                                flush=True,
                            )
                            return
                    elif (
                        step % options["save_every_steps"] == 0
                        or step == full_steps
                        or stop["signal"]
                    ):
                        save(epoch_number, step, loss_sum)
                    if stop["signal"]:
                        raise SystemExit(128 + stop["signal"])
                del progress, loader
            if args.smoke_test:
                raise ValueError("Smoke test requires at least two full batches")
            checkpoint = output / f"checkpoint_epoch_{epoch_number:03d}.pt"
            if not checkpoint.exists():
                link_checkpoint(last, checkpoint)
            val_directory = output / "cirr_val" / f"epoch_{epoch_number:03d}"
            evaluation_identity(val_directory, checkpoint, config)
            metrics_path = output / f"val_epoch_{epoch_number:03d}.json"
            if metrics_path.exists():
                metrics = json.loads(metrics_path.read_text())
            else:
                metrics = evaluate_cirr(model, processor, config, device, val_directory)
                write_json_atomic(metrics_path, metrics)
            score = sum(v for k, v in metrics.items() if "recall_at_" in k)
            if score >= best:
                best = score
                link_checkpoint(checkpoint, output / "best.pt")
                write_json_atomic(output / "best_metrics.json", {"epoch": epoch_number, **metrics})
            print(f"[epoch {epoch_number}] {json.dumps(metrics)}", flush=True)
            if stop["signal"]:
                raise SystemExit(128 + stop["signal"])
            next_step, loss_sum = 0, 0.0
        write_json_atomic(
            output / "training_complete.json",
            {
                "method": config["method"],
                "epochs": options["epochs"],
                "global_step": global_step,
                "main_table_checkpoint": str(checkpoint),
            },
        )
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def main():
    args = parse_args()
    config = load_config(args.config)
    ids, _, records = cirr_split(config, "train")
    val_ids, _, val_records = cirr_split(config, "val")
    minimum_batches = 2 if args.smoke_test else 1
    if len(records) < minimum_batches * config["training"]["batch_size"]:
        raise ValueError(f"This run needs at least {minimum_batches} full batches")
    model_path = Path(config["model"]["pretrained_model_name_or_path"])
    for filename in ("config.json", "vocab.json", "merges.txt"):
        if not (model_path / filename).is_file():
            raise FileNotFoundError(model_path / filename)
    if not any(
        (model_path / filename).is_file() for filename in ("pytorch_model.bin", "model.safetensors")
    ):
        raise FileNotFoundError(f"Local CLIP pretrained weights missing: {model_path}")
    print(
        json.dumps(
            {
                "config": config,
                "train_queries": len(records),
                "train_gallery": len(ids),
                "val_queries": len(val_records),
                "val_gallery": len(val_ids),
                "steps_per_epoch": len(records) // config["training"]["batch_size"],
                "model": model_label(config),
                "initialization": "public CLIP backbone only; no TG-CIR checkpoint",
            },
            indent=2,
        ),
        flush=True,
    )
    if args.dry_run:
        return
    if args.smoke_test:
        run(config, records, args)
        return
    output = Path(config["output_dir"])
    with output_lock(output):
        effective = output / "effective_config.json"
        if effective.exists() and training_identity(
            json.loads(effective.read_text())
        ) != training_identity(config):
            raise ValueError("Output belongs to a different training configuration")
        if not (output / "last.pt").is_file() and any(output.glob("checkpoint_epoch_*.pt")):
            raise RuntimeError(
                "Epoch checkpoints exist but last.pt is missing; refusing to restart over them"
            )
        write_json_atomic(effective, config)
        run(config, records, args)


if __name__ == "__main__":
    main()
