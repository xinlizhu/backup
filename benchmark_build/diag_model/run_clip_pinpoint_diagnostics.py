#!/usr/bin/env python3
"""Compare the trained CLIP-L baseline and hard-negative checkpoint on PinPoint.

This is intentionally a metric-only runner. It performs embedding extraction,
exact full-corpus retrieval, and deterministic metric aggregation; it never
calls the legacy scikit-learn probe pipeline.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from trained_clip_adapter import FROZEN_CLIP_MODELS, TRAINED_CLIP_MODELS


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_INPUT = (
    SCRIPT_DIR.parent
    / "cir_qdiag_openai"
    / "outputs"
    / "pinpoint_pilot_500_v7"
    / "final.jsonl"
)
DEFAULT_MODELS = "CLIP-L-baseline,CLIP-L-hard-w0.5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run metric-only PinPoint full-corpus diagnostics for the trained "
            "CLIP-L baseline and hard-negative-weight=0.5 models."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--pinpoint-root",
        type=Path,
        default=WORKSPACE_ROOT / "pinpoint-dataset",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--search-device", default=None)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--prefetch-batches", type=int, default=4)
    parser.add_argument("--processor-workers", type=int, default=2)
    parser.add_argument("--processor-prefetch-batches", type=int, default=2)
    parser.add_argument("--flush-interval", type=int, default=16)
    parser.add_argument("--query-search-batch-size", type=int, default=256)
    parser.add_argument("--corpus-search-chunk-size", type=int, default=8192)
    parser.add_argument("--ks", default="1,5,10,50")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-groups", type=int, default=None)
    parser.add_argument("--limit-index", type=int, default=None)
    parser.add_argument("--force-corpus", action="store_true")
    parser.add_argument("--force-queries", action="store_true")
    parser.add_argument("--force-search", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("[run] " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def write_combined_summaries(output_root: Path, models: list[str]) -> None:
    negative_frames: list[pd.DataFrame] = []
    positive_rows: list[dict] = []
    for model_name in models:
        analysis_dir = output_root / model_name / "analysis"
        negative = pd.read_csv(analysis_dir / "negative_summary.csv")
        negative.insert(0, "model", model_name)
        negative_frames.append(negative)
        positive_rows.append(
            json.loads((analysis_dir / "positive_summary.json").read_text(encoding="utf-8"))
        )

    negative_combined = pd.concat(negative_frames, ignore_index=True)
    negative_combined.to_csv(output_root / "negative_metrics_summary.csv", index=False)
    pd.DataFrame(positive_rows).to_csv(
        output_root / "positive_retrieval_summary.csv",
        index=False,
    )

    compact_columns = [
        "model",
        "subset",
        "group_count",
        "negative_count",
        "pair_balanced_mean_target_delta",
        "pair_balanced_target_raw_win_rate",
        "pair_balanced_target_robust_win_rate",
    ]
    for k in (1, 5, 10, 50):
        for stem in (
            "positive_recall",
            "negative_residual_recall",
            "delta_recall",
            "recall_win",
            "positive_ndcg",
            "negative_residual_ndcg",
            "delta_ndcg",
            "ndcg_win",
        ):
            compact_columns.append(f"pair_balanced_{stem}@{k}")
    compact_columns = [
        column for column in compact_columns if column in negative_combined.columns
    ]
    negative_combined.loc[
        negative_combined["subset"].isin(["all", "strict"]),
        compact_columns,
    ].to_csv(output_root / "comparison.csv", index=False)


def main() -> None:
    args = parse_args()
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    unknown = sorted(set(models) - set(TRAINED_CLIP_MODELS))
    if unknown:
        raise ValueError(f"Unsupported trained CLIP models: {unknown}")
    if not models:
        raise ValueError("--models selected no models")
    input_path = args.input.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_root = (
        args.output_root
        or input_path.parent / "clip_l_pinpoint_diagnostics"
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    shared_corpus_cache = output_root / "_shared_frozen_clip_l_corpus"

    forced_frozen_corpus = False
    for model_name in models:
        corpus_cache = (
            shared_corpus_cache
            if model_name in FROZEN_CLIP_MODELS
            else output_root / model_name / "cache" / "corpus"
        )
        command = [
            sys.executable,
            str(SCRIPT_DIR / "pinpoint_full_diagnostics.py"),
            "--input",
            str(input_path),
            "--model-name",
            model_name,
            "--model-path",
            str(TRAINED_CLIP_MODELS[model_name]),
            "--output-dir",
            str(output_root / model_name),
            "--corpus-cache-dir",
            str(corpus_cache),
            "--pinpoint-root",
            str(args.pinpoint_root.resolve()),
            "--image-cache",
            str((args.pinpoint_root / "image_cache").resolve()),
            "--index-signatures",
            str((args.pinpoint_root / "index_signatures.txt").resolve()),
            "--device",
            args.device,
            "--dtype",
            args.dtype,
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--prefetch-batches",
            str(args.prefetch_batches),
            "--processor-workers",
            str(args.processor_workers),
            "--processor-prefetch-batches",
            str(args.processor_prefetch_batches),
            "--flush-interval",
            str(args.flush_interval),
            "--query-search-batch-size",
            str(args.query_search_batch_size),
            "--corpus-search-chunk-size",
            str(args.corpus_search_chunk_size),
            "--ks",
            args.ks,
            "--bootstrap-replicates",
            str(args.bootstrap_replicates),
            "--seed",
            str(args.seed),
        ]
        if args.search_device:
            command.extend(["--search-device", args.search_device])
        if args.limit_groups is not None:
            command.extend(["--limit-groups", str(args.limit_groups)])
        if args.limit_index is not None:
            command.extend(["--limit-index", str(args.limit_index)])
        # Only frozen checkpoints may share candidate embeddings. Fine-tuned
        # checkpoints each get an independent corpus cache.
        if args.force_corpus:
            if model_name not in FROZEN_CLIP_MODELS or not forced_frozen_corpus:
                command.append("--force-corpus")
            if model_name in FROZEN_CLIP_MODELS:
                forced_frozen_corpus = True
        if args.force_queries:
            command.append("--force-queries")
        if args.force_search:
            command.append("--force-search")
        if args.analysis_only:
            command.append("--analysis-only")
        run(command)

    write_combined_summaries(output_root, models)
    print(f"[done] metric-only comparison: {output_root / 'comparison.csv'}")


if __name__ == "__main__":
    main()
