"""Qwen3-VL-2B-Instruct, shared all-linear LoRA, four controlled E4 runs."""
# ruff: noqa: E402 -- bootstrap the local package before importing the runner

import argparse
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))
# Subprocesses use the same interpreter and must see the same local source tree.
os.environ["PYTHONPATH"] = str(PROJECT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")

import run_main_comparison_e4 as suite

RUNS = (
    "qwen2_shared_oprd_s2",
    "qwen2_shared_random_kl_s2",
    "qwen2_shared_infonce",
    "qwen2_shared_cmse_s2",
)
DIRECT_RUNS = tuple(name.replace("qwen2_shared_", "qwen2_shared_direct_") for name in RUNS)


def main(*, direct=False):
    runs = DIRECT_RUNS if direct else RUNS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-indices", nargs="+", type=int, default=list(range(len(RUNS))))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--skip-extended-eval", action="store_true")
    args = parser.parse_args()
    if len(set(args.run_indices)) != len(args.run_indices) or any(i not in range(len(RUNS)) for i in args.run_indices):
        parser.error("run indices must be unique integers from 0 to 3")
    suite.RUNS = runs
    suite.ROOT = suite.PROJECT / ("outputs/qwen_shared_lora_direct_e4" if direct else "outputs/qwen_shared_lora_e4")
    suite.runner.ROOT = suite.ROOT
    suite.runner.LABEL_PREFIX = "SHARED-DIRECT" if direct else "SHARED"
    suite.runner.CONFIG_PATHS = {n: suite.PROJECT / f"configs/{n}.yaml" for n in runs}
    os.environ.setdefault("QWEN2_EVAL_BATCH_SIZE", "256" if direct else "4")
    if args.summarize_only:
        suite.summarize()
        return
    for i in args.run_indices:
        config = suite.load(runs[i])
        print(f"[{i}] {runs[i]}: {config.objective.loss_type}; candidates={config.mining.candidate_source}; "
              f"shared all-linear r={config.model.lora_rank}; filter={config.objective.teacher_target_max_rank}; "
              f"batch={config.optimizer.batch_size_per_gpu} x {config.optimizer.gradient_accumulation_steps}; "
              f"candidate_chunk={config.optimizer.candidate_encode_batch_size}; gradient_cache={config.optimizer.candidate_gradient_cache}; "
              f"output={config.experiment.output_dir}", flush=True)
        if not args.dry_run:
            suite.preflight(config)
    if args.dry_run:
        return
    for i in args.run_indices:
        if direct:
            suite.run_qwen(runs[i], extended=not args.skip_extended_eval, clip_style=True)
        else:
            suite.run_qwen(runs[i], extended=not args.skip_extended_eval)
        suite.summarize()


if __name__ == "__main__":
    main()
