#!/usr/bin/env python3
"""TG-CIR: train E4/B32, then evaluate the final epoch across all four suites."""

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from cir_qdiag_train.tg_cir_config import DEFAULT_CONFIG, load_config, model_label


def commands(config_path, config, device, training_only=False, eval_only=False):
    py = sys.executable
    output = Path(config["output_dir"])
    epoch = config["training"]["epochs"]
    final = output / f"checkpoint_epoch_{epoch:03d}.pt"
    label = f"{model_label(config)}-E{epoch}"
    if not eval_only:
        yield [py, PROJECT / "scripts/train_tg_cir.py", "--config", config_path, "--device", device]
    if training_only:
        return
    yield [
        py,
        PROJECT / "scripts/evaluate_tg_cir.py",
        "--checkpoint",
        final,
        "--dataset",
        "cirr",
        "--output-dir",
        output / "cirr_val" / f"epoch_{epoch:03d}",
        "--device",
        device,
    ]
    yield [
        py,
        PROJECT / "scripts/evaluate_tg_cir.py",
        "--checkpoint",
        final,
        "--dataset",
        "fashioniq",
        "--output-dir",
        output / "fashioniq" / f"epoch_{epoch:03d}",
        "--device",
        device,
    ]
    diag = PROJECT.parent / "diag_model"
    common = [
        "--adapter",
        f"{label}={final}",
        "--device",
        device,
        "--batch-size",
        "128",
        "--num-workers",
        "8",
        "--prefetch-batches",
        "2",
    ]
    yield [
        py,
        diag / "run_clip_pinpoint_official.py",
        "--models",
        label,
        *common,
        "--output-root",
        output / "pinpoint_official",
    ]
    yield [
        py,
        diag / "evaluate_winrate_random10.py",
        *common,
        "--output-root",
        output / "qdiag",
        "--random-samples",
        "10",
        "--seed",
        "42",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--train-only", action="store_true")
    group.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    print(
        json.dumps(
            {
                "epochs": config["training"]["epochs"],
                "batch_size": config["training"]["batch_size"],
                "output": config["output_dir"],
            }
        ),
        flush=True,
    )
    output = Path(config["output_dir"])
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)
    # Lock the complete pipeline as well as the trainer, preventing duplicate eval caches.
    if args.dry_run:
        for command in commands(
            args.config.resolve(), config, args.device, args.train_only, args.eval_only
        ):
            print(shlex.join(map(str, command)), flush=True)
        return
    import fcntl

    with (output / ".pipeline.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another TG-CIR pipeline is already running") from error
        for command in commands(
            args.config.resolve(), config, args.device, args.train_only, args.eval_only
        ):
            print("[command] " + shlex.join(map(str, command)), flush=True)
            subprocess.run(
                list(map(str, command)),
                cwd=PROJECT,
                check=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )


if __name__ == "__main__":
    main()
