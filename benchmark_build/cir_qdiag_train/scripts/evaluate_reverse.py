from __future__ import annotations

import argparse
import json
from pathlib import Path

from cir_qdiag_train.config import load_config
from cir_qdiag_train.reverse_evaluation import evaluate_cirr_reverse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate CIRR target-image to composed-query reverse retrieval."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics = evaluate_cirr_reverse(
        load_config(args.config),
        checkpoint_path=args.checkpoint,
        output_path=args.output,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
