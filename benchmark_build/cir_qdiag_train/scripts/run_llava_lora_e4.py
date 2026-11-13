"""LLaVA-NeXT all-linear LoRA: five isolated retrieval comparisons."""
import argparse
import fcntl
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))
os.environ["PYTHONPATH"] = str(PROJECT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
import run_main_comparison_e4 as suite

RUNS = ("llava_oprd_s2", "llava_oprd_qwen8", "llava_infonce", "llava_cmse_s2", "llava_cmse_qwen8")
ROOT = PROJECT / "outputs/llava_lora_2gpu_b32_gc8_e2"


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
    suite.ROOT = suite.runner.ROOT = ROOT
    suite.runner.LABEL_PREFIX = "LLAVA"
    suite.runner.CONFIG_PATHS = {n: PROJECT / f"configs/{n}.yaml" for n in RUNS}
    os.environ["QWEN2_EVAL_BATCH_SIZE"] = "16"
    os.environ["QWEN2_DIAG_BATCH_SIZE"] = "16"
    if args.summarize_only:
        suite.summarize()
        return
    for i in args.run_indices:
        c = suite.load(RUNS[i])
        if not c.model.llava:
            raise ValueError("Use LLaVA configs with this runner")
        print(f"[{i}] {RUNS[i]}: {c.objective.loss_type}; {c.mining.candidate_source}; "
              f"{c.model.lora_target_modules} r={c.model.lora_rank}; both sides; no fixed gallery; "
              f"batch={c.optimizer.batch_size_per_gpu} x {c.optimizer.gradient_accumulation_steps}; "
              f"GPUs={c.optimizer.world_size}; global_batch={c.optimizer.batch_size_per_gpu * c.optimizer.world_size * c.optimizer.gradient_accumulation_steps}; "
              f"K={c.mining.top_k}; candidate_chunk={c.optimizer.candidate_encode_batch_size}; "
              f"gradient_cache={c.optimizer.candidate_gradient_cache}; "
              f"workers_per_rank={c.data.num_workers}; prefetch={c.data.prefetch_factor}; "
              f"epochs={c.optimizer.epochs}; output={c.experiment.output_dir}", flush=True)
        if not args.dry_run:
            suite.preflight(c)
            if not (Path(c.model.pretrained_model_name_or_path) / "model.safetensors.index.json").is_file():
                raise FileNotFoundError("Run scripts/download_llava.py first")
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
