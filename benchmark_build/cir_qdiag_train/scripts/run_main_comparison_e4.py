"""Nine missing main-table cells; random baselines and one reverse-KL OPRD."""

import argparse
import csv
import fcntl
import importlib.util
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cir_qdiag_train.on_policy_config import load_on_policy_config
from cir_qdiag_train.qwen_on_policy_config import load_qwen_on_policy_config
from cir_qdiag_train.resume_config import without_loader_settings

PROJECT = Path(__file__).resolve().parent.parent
ROOT = PROJECT / "outputs/main_comparison_e4"
RUNS = (
    "clip_l_infonce",
    "clip_b_infonce",
    "qwen2_infonce",
    "clip_l_cmse_qwen8",
    "clip_b_cmse_s2",
    "clip_b_cmse_qwen8",
    "qwen2_cmse_s2",
    "qwen2_cmse_qwen8",
    "qwen2_oprd_s2",
)
spec = importlib.util.spec_from_file_location(
    "main9_runner", PROJECT / "scripts/run_reverse_rank_r128.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.ROOT = ROOT
runner.LABEL_PREFIX = "MAIN9"
runner.CONFIG_PATHS = {name: PROJECT / f"configs/main9_{name}.yaml" for name in RUNS}


def load(name):
    loader = load_qwen_on_policy_config if name.startswith(("qwen2", "qwen35", "llava")) else load_on_policy_config
    return loader(runner.CONFIG_PATHS[name])


def preflight(config):
    caches = []
    if config.objective.needs_teacher_scores:
        caches.append(config.data.teacher_cache_dir)
    if getattr(config.data, "student_gallery_cache_dir", None) is not None and not config.model.train_gallery:
        caches.append(config.data.student_gallery_cache_dir)
    for cache in caches:
        if not runner.read_json(cache / "metadata.json").get("complete"):
            raise RuntimeError(
                f"Missing/incomplete shared cache: {cache}; prepare it before training"
            )


# Explicit evaluation-only selections; do not change a checkpoint's training schedule.
EVALUATION_EPOCHS = {}


def execute_evaluation_jobs(jobs, log, evaluation_root, *, parallel=False):
    """Run independent evaluation lanes; join both before returning or raising."""
    if not parallel:
        for command, cwd in jobs:
            runner.execute(command, log, cwd=cwd)
        return
    lanes = [[], []]
    for command, cwd in jobs:
        command = list(command)
        lane = int(Path(command[1]).name in {
            "run_clip_pinpoint_official.py", "evaluate_winrate_random10.py",
        })
        command[command.index("--device") + 1] = f"cuda:{lane}"
        lanes[lane].append((command, cwd))
    evaluation_root.mkdir(parents=True, exist_ok=True)

    def run_lane(lane):
        with (evaluation_root / f"evaluation_gpu{lane}.log").open("a", buffering=1) as lane_log:
            for command, cwd in lanes[lane]:
                runner.execute(command, lane_log, cwd=cwd)

    if any(lanes):
        log.write("[evaluation] GPU 0: CIRR/FashionIQ; GPU 1: PinPoint/QDiag; logs: evaluation_gpu{0,1}.log\n")
        log.flush()
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run_lane, lane) for lane in range(2) if lanes[lane]]
            for future in futures:
                future.result()


def run_qwen(name, extended=True, *, clip_style=False):
    config = load(name)
    config_path = runner.CONFIG_PATHS[name]
    run = config.experiment.output_dir
    run.mkdir(parents=True, exist_ok=True)
    with (run / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = runner.read_json(run / "effective_config.json")
        if previous:
            from cir_qdiag_train.qwen_on_policy_config import QwenOnPolicyDistillationConfig
            previous = QwenOnPolicyDistillationConfig.model_validate(previous).model_dump(mode="json")
        if previous and without_loader_settings(previous) != without_loader_settings(config.model_dump(mode="json")):
            if name.startswith("llava"):
                from cir_qdiag_train.resume_config import restart_llava_attention
                if restart_llava_attention(previous, config.model_dump(mode="json"), run):
                    previous = None
        if previous and without_loader_settings(previous) != without_loader_settings(config.model_dump(mode="json")):
            raise RuntimeError(f"Refusing changed configuration in {run}")
        with (run / "runner.log").open("a", buffering=1) as log:
            py = sys.executable
            selected_epoch = EVALUATION_EPOCHS.get(name)
            final_epoch = config.optimizer.epochs if selected_epoch is None else selected_epoch
            if not 1 <= final_epoch <= config.optimizer.epochs:
                raise ValueError(f"Invalid selected epoch: {final_epoch}")
            final = run / f"checkpoint_epoch_{final_epoch:03d}.pt"
            if selected_epoch is not None and not final.is_file():
                raise FileNotFoundError(f"Evaluation-only checkpoint missing: {final}; training will not restart")
            evaluation_root = run if selected_epoch is None else run / f"selected_epoch_{final_epoch:03d}"
            if not final.is_file():
                command = [
                    py,
                    PROJECT / ("scripts/train_llava_on_policy_distillation.py" if config.model.llava
                               else "scripts/train_qwen_on_policy_distillation.py"),
                    "--config",
                    config_path,
                ]
                if config.optimizer.world_size > 1:
                    command = [py, "-m", "torch.distributed.run", "--standalone",
                               f"--nproc_per_node={config.optimizer.world_size}",
                               "--max_restarts=0", *command[1:]]
                if (run / "last.pt").is_file():
                    command += ["--resume-from", run / "last.pt"]
                runner.execute(command, log)
            if not final.is_file():
                raise FileNotFoundError(final)
            label = f"{runner.LABEL_PREFIX}-{name}"
            batch = os.environ.get("QWEN2_EVAL_BATCH_SIZE", str(config.evaluation.batch_size) if clip_style else "16")
            diag_batch = os.environ.get("QWEN2_DIAG_BATCH_SIZE", "128" if clip_style else batch)
            evaluation_jobs = []

            def queue_evaluation(command, _log, cwd=PROJECT):
                evaluation_jobs.append((command, cwd))

            # Evaluate the configured final epoch, or all epochs for CLIP-style runs.
            for epoch in (range(1, final_epoch + 1) if clip_style else (final_epoch,)):
                checkpoint = run / f"checkpoint_epoch_{epoch:03d}.pt"
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                epoch_label = label if epoch == final_epoch else f"{label}-E{epoch:03d}"
                cirr = evaluation_root / "cirr_val" if epoch == final_epoch else evaluation_root / "cirr_val" / f"epoch_{epoch:03d}"
                if not (cirr / "val_metrics.json").is_file():
                    queue_evaluation(
                        [
                            py,
                            PROJECT / "scripts/evaluate_cirr_original_adapter.py",
                            "--config",
                            config_path,
                            "--model-name",
                            epoch_label,
                            "--model-path",
                            checkpoint,
                            "--output-dir",
                            cirr,
                            "--device",
                            "cuda:0",
                            "--dtype",
                            "auto",
                            "--image-batch-size",
                            batch,
                            "--query-batch-size",
                            batch,
                        ],
                        log,
                    )
                fashion = evaluation_root / "fashioniq" if epoch == final_epoch else evaluation_root / "fashioniq" / f"epoch_{epoch:03d}"
                if not (fashion / "metrics.json").is_file():
                    queue_evaluation(
                        [
                            py,
                            PROJECT / "scripts/evaluate_fashioniq_original_adapter.py",
                            "--model-name",
                            epoch_label,
                            "--adapter-name",
                            epoch_label,
                            "--model-path",
                            checkpoint,
                            "--dataset-root",
                            os.environ.get(
                                "FASHIONIQ_ROOT",
                                str(PROJECT.parent.parent / "CIR_new/datasets/fashion-iq"),
                            ),
                            "--output-dir",
                            fashion,
                            "--device",
                            "cuda:0",
                            "--image-batch-size",
                            batch,
                            "--query-batch-size",
                            batch,
                        ],
                        log,
                    )
            if extended:
                diag = PROJECT.parent / "diag_model"
                official = evaluation_root / "pinpoint_official"
                if not (official / label / "official_metrics.csv").is_file():
                    queue_evaluation(
                        [
                            py,
                            diag / "run_clip_pinpoint_official.py",
                            "--models",
                            label,
                            "--adapter",
                            f"{label}={final}",
                            "--output-root",
                            official,
                            "--device",
                            "cuda:0",
                            "--batch-size",
                            diag_batch,
                        ],
                        log,
                        cwd=diag,
                    )
                qdiag = evaluation_root / "qdiag"
                if not (qdiag / label / "summary.csv").is_file():
                    queue_evaluation(
                        [
                            py,
                            diag / "evaluate_winrate_random10.py",
                            "--adapter",
                            f"{label}={final}",
                            "--output-root",
                            qdiag,
                            "--device",
                            "cuda:0",
                            "--batch-size",
                            diag_batch,
                            "--random-samples",
                            "10",
                            "--seed",
                            "42",
                        ],
                        log,
                        cwd=diag,
                    )

            execute_evaluation_jobs(
                evaluation_jobs, log, evaluation_root,
                parallel=(config.model.variant in {"qwen3_vl_embedding_shared_lora", "qwen3_vl_shared_lora"}
                          and config.optimizer.world_size == 2),
            )


def flatten(prefix, value, row):
    for key, item in value.items():
        if isinstance(item, dict):
            flatten(prefix + key + "/", item, row)
        elif isinstance(item, (int, float)):
            row[prefix + key] = item


def summarize():
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / ".summary.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = []
        for name in RUNS:
            run = load(name).experiment.output_dir
            if name in EVALUATION_EPOCHS:
                run = run / f"selected_epoch_{EVALUATION_EPOCHS[name]:03d}"
            qwen = name.startswith(("qwen2", "qwen35", "llava"))
            label = f"{runner.LABEL_PREFIX}-{name}"
            cirr = runner.read_json(
                run / ("cirr_val/val_metrics.json" if qwen else "val_epoch_004.json")
            )
            fashion = runner.read_json(
                run / ("fashioniq/metrics.json" if qwen else "fashioniq/epoch_004/metrics.json")
            )
            pp = runner.read_csv(run / "pinpoint_official" / label / "official_metrics.csv")
            qpath = run / "qdiag" / label / "summary.csv"
            row = {"Run": name, "Complete": bool(cirr and fashion and pp and qpath.is_file())}
            flatten("CIRR/", cirr, row)
            flatten("FashionIQ/", fashion, row)
            row.update({"PinPoint/" + k: v for k, v in pp.items() if k != "model"})
            if qpath.is_file():
                with qpath.open() as handle:
                    for q in csv.DictReader(handle):
                        row.update(
                            {f"QDiag/{q['subset']}/{k}": v for k, v in q.items() if k != "subset"}
                        )
            rows.append(row)
        fields = list(dict.fromkeys(k for row in rows for k in row))
        temp = ROOT / "all_metrics.csv.tmp"
        with temp.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temp.replace(ROOT / "all_metrics.csv")
        temp = ROOT / "all_metrics.json.tmp"
        temp.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
        temp.replace(ROOT / "all_metrics.json")
    print(ROOT / "all_metrics.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-indices", nargs="+", type=int, default=list(range(9)))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--skip-extended-eval", action="store_true")
    args = parser.parse_args()
    if len(set(args.run_indices)) != len(args.run_indices) or any(
        i not in range(9) for i in args.run_indices
    ):
        parser.error("run indices must be unique integers from 0 to 8")
    if args.summarize_only:
        summarize()
        return
    for index in args.run_indices:
        name = RUNS[index]
        config = load(name)
        print(
            f"[{index}] {name}: {config.objective.loss_type}, candidates={config.mining.candidate_source}, K=32, teacher_filter={config.objective.teacher_target_max_rank}"
        )
        if not args.dry_run:
            preflight(config)
    if args.dry_run:
        return
    for index in args.run_indices:
        name = RUNS[index]
        if name.startswith("qwen2"):
            run_qwen(name, extended=not args.skip_extended_eval)
        else:
            runner.run_one(name, extended=not args.skip_extended_eval)
        summarize()


if __name__ == "__main__":
    main()
