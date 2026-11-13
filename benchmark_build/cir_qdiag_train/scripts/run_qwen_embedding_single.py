"""Independent single-GPU Embedding experiments, matched to Instruct configs."""
import argparse
import fcntl
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
os.environ["PYTHONPATH"] = str(PROJECT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
import run_main_comparison_e4 as suite
from cir_qdiag_train.qwen_on_policy_config import load_qwen_on_policy_config

SUFFIXES = ("oprd_s2", "oprd_qwen8", "infonce", "cmse_s2", "cmse_qwen8")
RUNS = tuple("qwen2_embedding_single_" + s for s in SUFFIXES)
ROOT = PROJECT / "outputs/qwen_embedding_single"


def validate_matched(config, suffix):
    old = load_qwen_on_policy_config(PROJECT / f"configs/qwen2_instruct_shared_{suffix}.yaml")
    actual, expected = config.model_dump(mode="json"), old.model_dump(mode="json")
    expected["experiment"] = actual["experiment"]
    expected["model"]["variant"] = "qwen3_vl_embedding_shared_lora"
    expected["model"]["pretrained_model_name_or_path"] = str(PROJECT.parents[1] / "CIR_PLAY/models/Qwen3-VL-Embedding-2B")
    if actual != expected or config.optimizer.world_size != 1:
        raise ValueError("Single-GPU configuration no longer matches its Instruct reference")
    if config.data.student_gallery_cache_dir is not None:
        raise ValueError("Fixed student gallery forbidden")


def summarize_locked():
    ROOT.mkdir(parents=True, exist_ok=True)
    # Serialize only summary writes, never the independent experiments.
    with (ROOT / ".summary.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        suite.summarize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-indices", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--skip-extended-eval", action="store_true")
    args = parser.parse_args()
    if len(set(args.run_indices)) != len(args.run_indices) or any(i not in range(5) for i in args.run_indices):
        parser.error("run indices must be unique integers from 0 to 4")
    suite.RUNS = RUNS
    suite.EVALUATION_EPOCHS = {}
    suite.ROOT = suite.runner.ROOT = ROOT
    suite.runner.LABEL_PREFIX = "EMBEDDING-SINGLE"
    suite.runner.CONFIG_PATHS = {n: PROJECT / f"configs/{n}.yaml" for n in RUNS}
    os.environ["QWEN2_EVAL_BATCH_SIZE"] = "16"
    os.environ["QWEN2_DIAG_BATCH_SIZE"] = "16"
    if args.summarize_only:
        summarize_locked()
        return
    for i in args.run_indices:
        c = suite.load(RUNS[i])
        validate_matched(c, SUFFIXES[i])
        print(f"[{i}] {RUNS[i]}: single GPU; batch=16 x 2; effective_batch=32; "
              f"all-linear r={c.model.lora_rank}; workers={c.data.num_workers}; "
              f"prefetch={c.data.prefetch_factor}; epochs={c.optimizer.epochs}; "
              f"KL={c.objective.rank_kl_direction}; output={c.experiment.output_dir}", flush=True)
        if not args.dry_run:
            suite.preflight(c)
    if args.dry_run:
        return
    # run_qwen holds each experiment's own .run.lock. No global suite lock:
    # different --run-indices may run concurrently on different visible GPUs.
    for i in args.run_indices:
        suite.run_qwen(RUNS[i], extended=not args.skip_extended_eval)
        summarize_locked()


if __name__ == "__main__":
    main()
