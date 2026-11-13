from __future__ import annotations

import argparse
import json
from pathlib import Path

from cir_qdiag_train.config import load_config
from cir_qdiag_train.evaluation import evaluate_cirr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument("--checkpoint", type=Path)
    checkpoint_group.add_argument(
        "--pretrained-only",
        action="store_true",
        help="Evaluate the original pretrained backbone with the zero residual composer.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    metrics = evaluate_cirr(
        load_config(args.config),
        checkpoint_path=None if args.pretrained_only else args.checkpoint,
        output_path=args.output,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
