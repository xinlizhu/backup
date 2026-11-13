from __future__ import annotations

import argparse
from pathlib import Path

from cir_qdiag_train.config import TrainConfig, load_config
from cir_qdiag_train.trainer import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--resume-from",
        type=Path,
        help="Resume model, optimizer, scheduler, epoch, and step from last.pt.",
    )
    checkpoint_group.add_argument(
        "--initialize-from",
        type=Path,
        help="Load model weights only and start a new optimizer/schedule at epoch 1.",
    )
    parser.add_argument("--experiment-name")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size-per-gpu", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--hard-negative-weight", type=float)
    parser.add_argument("--hard-negative-margin", type=float)
    parser.add_argument("--save-every-epochs", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    payload = config.model_dump()
    if args.experiment_name is not None:
        payload["experiment"]["name"] = args.experiment_name
    if args.output_dir is not None:
        payload["experiment"]["output_dir"] = args.output_dir.resolve()
    for argument, field in (
        (args.epochs, "epochs"),
        (args.batch_size_per_gpu, "batch_size_per_gpu"),
        (args.gradient_accumulation_steps, "gradient_accumulation_steps"),
        (args.hard_negative_weight, "hard_negative_weight"),
        (args.hard_negative_margin, "hard_negative_margin"),
        (args.save_every_epochs, "save_every_epochs"),
    ):
        if argument is not None:
            payload["optimizer"][field] = argument
    config = TrainConfig.model_validate(payload)
    train(
        config,
        resume_from=args.resume_from,
        initialize_from=args.initialize_from,
    )


if __name__ == "__main__":
    main()
