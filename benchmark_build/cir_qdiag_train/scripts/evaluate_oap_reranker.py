from __future__ import annotations

import argparse
import json
from pathlib import Path

from cir_qdiag_train.oap_config import load_oap_config
from cir_qdiag_train.oap_evaluation import evaluate_oap_reranker


def _float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def _int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--betas", type=_float_list, required=True)
    parser.add_argument("--top-ks", type=_int_list, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics = evaluate_oap_reranker(
        load_oap_config(args.config),
        checkpoint_path=args.checkpoint,
        betas=args.betas,
        rerank_top_ks=args.top_ks,
        output_path=args.output,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
