from __future__ import annotations

import argparse
import json
from pathlib import Path

from cir_qdiag_train.oap_config import load_oap_config
from cir_qdiag_train.oap_evaluation import evaluate_oap_cirr


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare frozen global CIRR retrieval with O/A/P semantic scoring."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--beta",
        type=float,
        help="Override semantic residual weight at inference without retraining.",
    )
    args = parser.parse_args()
    config = load_oap_config(args.config)
    if args.beta is not None:
        if args.beta <= 0:
            parser.error("--beta must be positive")
        config.model.beta = args.beta
    metrics = evaluate_oap_cirr(
        config,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
