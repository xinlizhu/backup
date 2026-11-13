#!/usr/bin/env python3
"""Run PinPoint V7 full-corpus diagnostics sequentially for Base and MLLM-S1."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]
CHECKPOINT_ROOT = WORKSPACE_ROOT / "CIR_PLAY" / "models" / "checkpoints"
MODEL_PATHS = {
    "BGE-VL-base": CHECKPOINT_ROOT / "BGE-VL-base",
    "BGE-VL-MLLM-S1": CHECKPOINT_ROOT / "BGE-VL-MLLM-S1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PinPoint V7 full-corpus diagnostics for two BGE models."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--models",
        default="BGE-VL-base,BGE-VL-MLLM-S1",
        help="Comma-separated subset of BGE-VL-base,BGE-VL-MLLM-S1.",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--pinpoint-root",
        type=Path,
        default=WORKSPACE_ROOT / "pinpoint-dataset",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--search-device", default=None)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--base-batch-size", type=int, default=1024)
    parser.add_argument("--mllm-batch-size", type=int, default=16)
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
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Also create PCA/t-SNE q+-q- vector diagnostics from cached queries.",
    )
    parser.add_argument(
        "--visualization-only",
        action="store_true",
        help="Only visualize existing PinPoint query caches; do not rerun retrieval analysis.",
    )
    parser.add_argument("--visualization-max-points", type=int, default=6000)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-iterations", type=int, default=1500)
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("[run] " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    models = [value.strip() for value in args.models.split(",") if value.strip()]
    unknown = sorted(set(models) - set(MODEL_PATHS))
    if unknown:
        raise ValueError(f"Unsupported models: {unknown}")
    if not models:
        raise ValueError("--models selected no models")
    input_path = args.input.resolve()
    output_root = (
        args.output_root
        or input_path.parent / "full_retrieval_diagnostics"
    ).resolve()
    for model_name in models:
        if not args.visualization_only:
            batch_size = (
                args.mllm_batch_size
                if model_name == "BGE-VL-MLLM-S1"
                else args.base_batch_size
            )
            command = [
                sys.executable,
                str(SCRIPT_DIR / "pinpoint_full_diagnostics.py"),
                "--input",
                str(input_path),
                "--model-name",
                model_name,
                "--model-path",
                str(MODEL_PATHS[model_name]),
                "--output-dir",
                str(output_root / model_name),
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
                str(batch_size),
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
            if args.force_corpus:
                command.append("--force-corpus")
            if args.force_queries:
                command.append("--force-queries")
            if args.force_search:
                command.append("--force-search")
            if args.analysis_only:
                command.append("--analysis-only")
            run(command)
        if args.visualize or args.visualization_only:
            run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "visualize_pinpoint_vectors.py"),
                    "--input",
                    str(input_path),
                    "--model-dir",
                    str(output_root / model_name),
                    "--output-dir",
                    str(output_root / model_name / "analysis" / "vector_visualization"),
                    "--max-points",
                    str(args.visualization_max_points),
                    "--perplexity",
                    str(args.tsne_perplexity),
                    "--tsne-iterations",
                    str(args.tsne_iterations),
                    "--seed",
                    str(args.seed),
                ]
            )

    combined: list[pd.DataFrame] = []
    positive_summaries: list[dict] = []
    for model_name in models:
        summary_path = output_root / model_name / "analysis" / "negative_summary.csv"
        summary = pd.read_csv(summary_path)
        summary.insert(0, "model", model_name)
        combined.append(summary)
        positive_path = output_root / model_name / "analysis" / "positive_summary.json"
        positive_summaries.append(json.loads(positive_path.read_text(encoding="utf-8")))
    output_root.mkdir(parents=True, exist_ok=True)
    pd.concat(combined, ignore_index=True).to_csv(
        output_root / "combined_negative_summary.csv",
        index=False,
    )
    (output_root / "combined_positive_summary.json").write_text(
        json.dumps(positive_summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[done] combined outputs={output_root}")


if __name__ == "__main__":
    main()
