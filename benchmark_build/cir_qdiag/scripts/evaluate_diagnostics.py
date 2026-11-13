from __future__ import annotations

import argparse
import json
from pathlib import Path

from cir_qdiag.evaluation.artifact_check import run_text_only_baseline
from cir_qdiag.evaluation.metrics import evaluate_scores
from cir_qdiag.evaluation.scoring import load_scorer, score_records
from cir_qdiag.schemas import SampleRecord
from cir_qdiag.utils.jsonl import read_jsonl, read_models, write_jsonl_atomic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="final.jsonl")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scorer", help="Python scorer as module:attribute")
    group.add_argument("--scores", type=Path, help="precomputed score JSONL")
    parser.add_argument("--scores-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-baseline", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    records = list(read_models(args.input, SampleRecord))
    if args.scorer:
        rows = score_records(records, load_scorer(args.scorer))
        if args.scores_output:
            write_jsonl_atomic(args.scores_output, rows)
    else:
        rows = list(read_jsonl(args.scores))
    report = evaluate_scores(rows)
    if args.artifact_baseline:
        report["text_only_artifact_baseline"] = run_text_only_baseline(records, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
