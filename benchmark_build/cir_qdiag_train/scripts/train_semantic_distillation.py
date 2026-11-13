#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from cir_qdiag_train.distillation_config import load_distillation_config
from cir_qdiag_train.distillation_trainer import train_semantic_distillation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    train_semantic_distillation(
        load_distillation_config(args.config),
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
