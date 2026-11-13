"""Five Embedding-2B two-GPU comparisons with shared all-linear LoRA and live candidates."""
# ruff: noqa: E402
import argparse
import fcntl
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))
os.environ["PYTHONPATH"] = str(PROJECT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")

import run_main_comparison_e4 as suite

RUNS = tuple("qwen2_embedding_shared_" + suffix for suffix in
             ("oprd_s2", "infonce", "cmse_s2", "oprd_qwen8", "cmse_qwen8"))
ROOT = PROJECT / "outputs/qwen_embedding_shared_2gpu_e1"


def validate(config):
    if config.model.variant != "qwen3_vl_embedding_shared_lora" or config.optimizer.world_size != 2:
        raise ValueError("Requires Embedding shared LoRA on two GPUs")
    if not config.model.shared_lora or config.model.lora_target_modules != "all-linear":
        raise ValueError("This suite requires shared all-linear LoRA")
    if config.data.student_gallery_cache_dir is not None:
        raise ValueError("A fixed student gallery is not allowed")
    if config.objective.uses_teacher_filter or not config.mining.force_target_insertion:
        raise ValueError("Use target insertion without teacher filtering")
    if config.mining.refresh_mode != "every_epoch":
        raise ValueError("Refresh candidates each epoch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-indices", nargs="+", type=int, default=list(range(len(RUNS))))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--skip-extended-eval", action="store_true")
    args = parser.parse_args()
    if len(set(args.run_indices)) != len(args.run_indices) or any(i not in range(len(RUNS)) for i in args.run_indices):
        parser.error("run indices must be unique integers from 0 to 4")
    suite.RUNS = RUNS
    suite.EVALUATION_EPOCHS = {}
    suite.ROOT = suite.runner.ROOT = ROOT
    suite.runner.LABEL_PREFIX = "EMBEDDING-SHARED"
    suite.runner.CONFIG_PATHS = {name: PROJECT / f"configs/{name}.yaml" for name in RUNS}
    os.environ["QWEN2_EVAL_BATCH_SIZE"] = "16"
    os.environ["QWEN2_DIAG_BATCH_SIZE"] = "16"
    if args.summarize_only:
        suite.summarize()
        return
    for i in args.run_indices:
        c = suite.load(RUNS[i])
        validate(c)
        print(f"[{i}] {RUNS[i]}: {c.objective.loss_type}; shared all-linear LoRA; "
              f"live candidates={c.mining.candidate_source}; no fixed gallery; "
              f"batch={c.optimizer.batch_size_per_gpu} x {c.optimizer.gradient_accumulation_steps}; "
              f"GPUs={c.optimizer.world_size}; global_batch={c.optimizer.batch_size_per_gpu * c.optimizer.world_size}; "
              f"candidate_chunk={c.optimizer.candidate_encode_batch_size}; epochs={c.optimizer.epochs}; "
              f"workers={c.data.num_workers}; filter=None; output={c.experiment.output_dir}", flush=True)
        if not args.dry_run:
            suite.preflight(c)
    if args.dry_run:
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / ".suite.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for i in args.run_indices:
            suite.run_qwen(RUNS[i], extended=not args.skip_extended_eval)
            suite.summarize()


if __name__ == "__main__":
    main()
