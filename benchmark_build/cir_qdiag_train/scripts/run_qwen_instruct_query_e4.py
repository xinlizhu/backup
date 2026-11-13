"""Original Qwen Embedding E4 protocol; only switch initialization to Instruct."""
# Make subprocesses use the same local source tree.

import argparse
import fcntl
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))
os.environ["PYTHONPATH"] = str(PROJECT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")

import run_main_comparison_e4 as suite

SOURCE_CONFIGS = {
    "oprd_s2": "main9_qwen2_oprd_s2",
    "oprd_qwen8": "opd_qwen2_qwen8_rank",
    "infonce": "main9_qwen2_infonce",
    "cmse_s2": "main9_qwen2_cmse_s2",
    "cmse_qwen8": "main9_qwen2_cmse_qwen8",
}
RUNS = tuple("qwen2_instruct_query_" + name for name in SOURCE_CONFIGS)
ROOT = PROJECT / "outputs/qwen_instruct_query_e4"


def configure_suite():
    suite.RUNS = RUNS
    suite.ROOT = suite.runner.ROOT = ROOT
    suite.runner.LABEL_PREFIX = "INSTRUCT-QUERY"
    suite.runner.CONFIG_PATHS = {n: PROJECT / f"configs/{n}.yaml" for n in RUNS}


def assert_only_initialization_changed(name, config):
    from cir_qdiag_train.qwen_on_policy_config import load_qwen_on_policy_config

    suffix = name.removeprefix("qwen2_instruct_query_")
    old = load_qwen_on_policy_config(PROJECT / f"configs/{SOURCE_CONFIGS[suffix]}.yaml")
    expected = old.model_dump(mode="json")
    expected["experiment"]["name"] = name
    expected["experiment"]["output_dir"] = str(ROOT / name)
    expected["data"]["student_gallery_cache_dir"] = str(ROOT / "cache/qwen3_vl_instruct_2b_fixed_gallery")
    expected["model"]["variant"] = "qwen3_vl_instruct_query_lora"
    expected["model"]["pretrained_model_name_or_path"] = str(PROJECT.parents[1] / "qw3/Qwen3-VL-2B-Instruct")
    if config.model_dump(mode="json") != expected:
        raise ValueError(f"{name}: configuration differs from the old protocol beyond initialization/output/cache identity")


def prepare_gallery(config_path, config):
    cache = config.data.student_gallery_cache_dir
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (cache / "prepare.log").open("a", buffering=1) as log:
            suite.runner.execute([
                sys.executable, PROJECT / "scripts/prepare_qwen_student_gallery_cache.py",
                "--config", config_path, "--image-batch-size", str(config.mining.image_batch_size),
            ], log)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-indices", nargs="+", type=int, default=list(range(len(RUNS))))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--skip-extended-eval", action="store_true")
    args = parser.parse_args()
    if (len(set(args.run_indices)) != len(args.run_indices)
            or any(i not in range(len(RUNS)) for i in args.run_indices)):
        parser.error("run indices must be unique integers from 0 to 4")
    configure_suite()
    # Do not inherit the large inference batches of the separate shared-LoRA job.
    os.environ["QWEN2_EVAL_BATCH_SIZE"] = "16"
    os.environ["QWEN2_DIAG_BATCH_SIZE"] = "16"
    if args.summarize_only:
        suite.summarize()
        return
    configs = []
    for i in args.run_indices:
        name = RUNS[i]
        c = suite.load(name)
        assert_only_initialization_changed(name, c)
        print(f"[{i}] {name}: {c.objective.loss_type}; query-only attention LoRA; "
              f"batch={c.optimizer.batch_size_per_gpu} x {c.optimizer.gradient_accumulation_steps}; "
              f"epochs={c.optimizer.epochs}; workers={c.data.num_workers}; prefetch={c.data.prefetch_factor}; "
              f"mining_batch={c.mining.query_batch_size}; filter={c.objective.teacher_target_max_rank}; "
              f"KL={c.objective.rank_kl_direction}; output={c.experiment.output_dir}", flush=True)
        configs.append(c)
    if args.dry_run:
        return
    # Fail before GPU work if a required teacher is unavailable.
    for c in configs:
        if (c.objective.needs_teacher_scores
                and not suite.runner.read_json(c.data.teacher_cache_dir / "metadata.json").get("complete")):
            raise RuntimeError(f"Missing teacher cache: {c.data.teacher_cache_dir}")
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / ".suite.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        first_name = RUNS[args.run_indices[0]]
        prepare_gallery(suite.runner.CONFIG_PATHS[first_name], configs[0])
        for i, c in zip(args.run_indices, configs, strict=True):
            suite.preflight(c)
            suite.run_qwen(RUNS[i], extended=not args.skip_extended_eval)
            suite.summarize()


if __name__ == "__main__":
    main()
