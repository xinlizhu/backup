#!/usr/bin/env python3
"""Run positive-paraphrase robustness sequentially for supported models."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from extract_embeddings import DEFAULT_BATCH_SIZES, SUPPORTED_MODELS


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run positive-paraphrase Delta calibration for VL models."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--paraphrases", required=True, type=Path)
    parser.add_argument("--diagnostics-root", required=True, type=Path)
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
    parser.add_argument("--interval-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("[run] " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    selected_models = [value.strip() for value in args.models.split(",") if value.strip()]
    unknown = sorted(set(selected_models) - set(SUPPORTED_MODELS))
    if unknown:
        raise ValueError(
            f"Unsupported models: {unknown}; supported={tuple(SUPPORTED_MODELS)}"
        )
    if not selected_models:
        raise ValueError("--models selected no models")

    diagnostics_root = args.diagnostics_root.resolve()
    for model_name in selected_models:
        if model_name == "BGE-VL-base":
            batch_size = args.base_batch_size
        elif model_name == "BGE-VL-MLLM-S1":
            batch_size = args.mllm_batch_size
        else:
            batch_size = args.qwen_batch_size
        command = [
            sys.executable,
            str(SCRIPT_DIR / "paraphrase_robustness.py"),
            "--input",
            str(args.input.resolve()),
            "--paraphrases",
            str(args.paraphrases.resolve()),
            "--diagnostics-dir",
            str(diagnostics_root / model_name),
            "--model-name",
            model_name,
            "--model-path",
            str(SUPPORTED_MODELS[model_name]),
            "--device",
            args.device,
            "--dtype",
            args.dtype,
            "--batch-size",
            str(batch_size),
            "--interval-level",
            str(args.interval_level),
            "--bootstrap-replicates",
            str(args.bootstrap_replicates),
            "--seed",
            str(args.seed),
        ]
        if args.force:
            command.append("--force")
        run(command)

    combined: list[pd.DataFrame] = []
    for model_name in selected_models:
        path = (
            diagnostics_root
            / model_name
            / "paraphrase_robustness"
            / "summary.csv"
        )
        frame = pd.read_csv(path)
        frame.insert(0, "model", model_name)
        combined.append(frame)
    output = diagnostics_root / "paraphrase_robustness_summary.csv"
    temporary = output.with_name(f".{output.name}.tmp")
    pd.concat(combined, ignore_index=True).to_csv(temporary, index=False)
    temporary.replace(output)
    print(f"[done] combined_summary={output}")


if __name__ == "__main__":
    main()
