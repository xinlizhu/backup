#!/usr/bin/env python3
"""Run extraction and offline analysis sequentially for supported VL models."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]
CHECKPOINT_ROOT = WORKSPACE_ROOT / "CIR_PLAY" / "models" / "checkpoints"
QWEN_MODEL_NAME = "Qwen3-VL-Embedding-2B"
MODEL_PATHS = {
    "BGE-VL-base": CHECKPOINT_ROOT / "BGE-VL-base",
    "BGE-VL-MLLM-S1": CHECKPOINT_ROOT / "BGE-VL-MLLM-S1",
    QWEN_MODEL_NAME: WORKSPACE_ROOT / "qw3" / QWEN_MODEL_NAME,
}
SUPPORTED_MODELS = tuple(MODEL_PATHS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CIR-QDiag diagnostics for VL models.")
    parser.add_argument("--input", required=True, type=Path, help="CIR-QDiag final.jsonl")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Defaults to <input-dir>/model_diagnostics.",
    )
    parser.add_argument(
        "--models",
        default=",".join(SUPPORTED_MODELS),
        help=f"Comma-separated subset of {','.join(SUPPORTED_MODELS)}.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--base-batch-size", type=int, default=2048)
    parser.add_argument("--mllm-batch-size", type=int, default=32)
    parser.add_argument("--qwen-batch-size", type=int, default=32)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Also run PCA/t-SNE vector visualization after the offline probe.",
    )
    parser.add_argument(
        "--visualization-only",
        action="store_true",
        help="Only visualize existing features.npz files; do not extract or retrain probes.",
    )
    parser.add_argument("--visualization-max-points", type=int, default=5000)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-iterations", type=int, default=1500)
    parser.add_argument(
        "--reference-ablation-only",
        action="store_true",
        help=(
            "Resume extraction to add f(R), then refresh only descriptive 3x3 "
            "and reference-ablation tables without retraining probes."
        ),
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("[run] " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    if args.extract_only and args.probe_only:
        raise ValueError("--extract-only and --probe-only cannot be used together")
    if args.visualization_only and (args.extract_only or args.probe_only):
        raise ValueError(
            "--visualization-only cannot be combined with --extract-only or --probe-only"
        )
    if args.extract_only and args.reference_ablation_only:
        raise ValueError(
            "--extract-only and --reference-ablation-only cannot be used together"
        )
    selected_models = [value.strip() for value in args.models.split(",") if value.strip()]
    unknown = sorted(set(selected_models) - set(SUPPORTED_MODELS))
    if unknown:
        raise ValueError(f"Unsupported models: {unknown}; supported={SUPPORTED_MODELS}")
    if not selected_models:
        raise ValueError("--models selected no models")
    input_path = args.input.resolve()
    output_root = (args.output_root or input_path.parent / "model_diagnostics").resolve()

    for model_name in selected_models:
        model_output = output_root / model_name
        if not args.probe_only and not args.visualization_only:
            if model_name == "BGE-VL-MLLM-S1":
                batch_size = args.mllm_batch_size
            elif model_name == QWEN_MODEL_NAME:
                batch_size = args.qwen_batch_size
            else:
                batch_size = args.base_batch_size
            extraction_command = [
                sys.executable,
                str(SCRIPT_DIR / "extract_embeddings.py"),
                "--input",
                str(input_path),
                "--model-name",
                model_name,
                "--model-path",
                str(MODEL_PATHS[model_name]),
                "--output-dir",
                str(model_output),
                "--device",
                args.device,
                "--dtype",
                args.dtype,
                "--batch-size",
                str(batch_size),
            ]
            if args.limit_samples is not None:
                extraction_command.extend(["--limit-samples", str(args.limit_samples)])
            if args.force:
                extraction_command.append("--force")
            run(extraction_command)

        if not args.extract_only and not args.visualization_only:
            analysis_command = [
                sys.executable,
                str(SCRIPT_DIR / "probe_grid.py"),
                "--features-dir",
                str(model_output),
                "--output-dir",
                str(model_output / "analysis"),
                "--folds",
                str(args.folds),
                "--bootstrap-replicates",
                str(args.bootstrap_replicates),
                "--seed",
                str(args.seed),
            ]
            if args.reference_ablation_only:
                analysis_command.append("--descriptive-only")
            run(analysis_command)

        if args.visualize or args.visualization_only:
            visualization_command = [
                sys.executable,
                str(SCRIPT_DIR / "visualize_vectors.py"),
                "--features-dir",
                str(model_output),
                "--output-dir",
                str(model_output / "analysis" / "vector_visualization"),
                "--max-points",
                str(args.visualization_max_points),
                "--perplexity",
                str(args.tsne_perplexity),
                "--tsne-iterations",
                str(args.tsne_iterations),
                "--seed",
                str(args.seed),
            ]
            run(visualization_command)


if __name__ == "__main__":
    main()
