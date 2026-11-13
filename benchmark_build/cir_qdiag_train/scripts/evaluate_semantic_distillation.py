#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cir_qdiag_train.distillation_config import load_distillation_config
from cir_qdiag_train.evaluation import evaluate_cirr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics = evaluate_cirr(
        load_distillation_config(args.config),
        checkpoint_path=args.checkpoint,
        output_path=args.output,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
