#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cir_qdiag_train.evaluation import evaluate_cirr
from cir_qdiag_train.on_policy_config import load_on_policy_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    args = parser.parse_args()
    metrics = evaluate_cirr(
        load_on_policy_config(args.config),
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
