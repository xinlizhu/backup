"""Paired Qwen3.5-0.8B OPRD runs: all-linear LoRA vs full backbone tuning."""

import argparse
import fcntl
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))
os.environ["PYTHONPATH"] = str(PROJECT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")

import run_main_comparison_e4 as suite

RUNS = ("qwen35_lora_oprd_s2", "qwen35_full_oprd_s2")
ROOT = PROJECT / "outputs/qwen35_e4"
CLIP_BATCH_RUNS = ("qwen35_clip_batch_lora_oprd_s2", "qwen35_clip_batch_full_oprd_s2")
CLIP_BATCH_ROOT = PROJECT / "outputs/qwen35_clip_batch_e4"


def main(*, clip_batch=False):
    # Keep the original entry point/configs intact for runs already in progress.
    runs = CLIP_BATCH_RUNS if clip_batch else RUNS
    root = CLIP_BATCH_ROOT if clip_batch else ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-indices", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--skip-extended-eval", action="store_true")
    args = parser.parse_args()
    if len(set(args.run_indices)) != len(args.run_indices) or any(i not in (0, 1) for i in args.run_indices):
        parser.error("run indices must be unique: 0=LoRA, 1=full")
    suite.RUNS = runs
    suite.ROOT = suite.runner.ROOT = root
    suite.runner.LABEL_PREFIX = "QWEN35-CLIP-BATCH" if clip_batch else "QWEN35"
    suite.runner.CONFIG_PATHS = {n: PROJECT / f"configs/{n}.yaml" for n in runs}
    os.environ["QWEN2_EVAL_BATCH_SIZE"] = "16"
    os.environ["QWEN2_DIAG_BATCH_SIZE"] = "16"
    if args.summarize_only:
        suite.summarize()
        return
    for i in args.run_indices:
        c = suite.load(runs[i])
        print(f"[{i}] {runs[i]}: {c.model.variant}; both sides; "
              f"batch={c.optimizer.batch_size_per_gpu} x {c.optimizer.gradient_accumulation_steps}; "
              f"candidate_chunk={c.optimizer.candidate_encode_batch_size}; "
              f"K={c.mining.top_k}; T={c.objective.rank_temperature}; "
              f"workers={c.data.num_workers}; prefetch={c.data.prefetch_factor}; output={c.experiment.output_dir}", flush=True)
        if not args.dry_run:
            suite.preflight(c)
    if args.dry_run:
        return
    try:
        from transformers import Qwen3_5ForConditionalGeneration
        del Qwen3_5ForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError("Use .venv_qwen35/bin/python; do not upgrade the old training environment") from exc
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".suite.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for i in args.run_indices:
            suite.run_qwen(runs[i], extended=not args.skip_extended_eval)
            suite.summarize()


if __name__ == "__main__":
    main()
