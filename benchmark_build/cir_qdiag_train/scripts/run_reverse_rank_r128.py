#!/usr/bin/env python3
"""Run five Reverse Rank controls, with disjoint logs/caches and resumable stages."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from cir_qdiag_train.on_policy_config import load_on_policy_config
from cir_qdiag_train.resume_config import without_loader_settings

PROJECT = Path(__file__).resolve().parent.parent
ROOT = PROJECT / "outputs/reverse_rank_r128_e4"
RUNS = ("static", "teacher", "random", "no_filter", "opd")
CONFIG_PATHS = {}
LABEL_PREFIX = "CLIP-L-REV"


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def read_csv(path, subset=None):
    if not path.is_file():
        return {}
    with path.open(newline="") as handle:
        return next(
            (r for r in csv.DictReader(handle) if subset is None or r.get("subset") == subset), {}
        )


def summarize():
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / ".summary.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = []
        for name in RUNS:
            run = ROOT / name
            label = f"CLIP-L-REV-{name}"
            pinpoint = read_csv(run / "pinpoint_official" / label / "official_metrics.csv")
            qdiag = read_csv(run / "qdiag" / label / "summary.csv", "strict")
            for epoch in range(1, 5):
                cirr = read_json(run / f"val_epoch_{epoch:03d}.json")
                fashion = read_json(run / "fashioniq" / f"epoch_{epoch:03d}" / "metrics.json")
                macro = fashion.get("macro_average", {})
                rows.append(
                    {
                        "Run": name,
                        "Epoch": epoch,
                        "Checkpoint": (run / f"checkpoint_epoch_{epoch:03d}.pt").is_file(),
                        "CIRR R@1": cirr.get("recall_at_1"),
                        "CIRR R@10": cirr.get("recall_at_10"),
                        "Group R@1": cirr.get("group_recall_at_1"),
                        "FashionIQ R@10": macro.get("recall_at_10"),
                        "FashionIQ R@50": macro.get("recall_at_50"),
                        "PinPoint mAP@10": pinpoint.get("mAP@10") if epoch == 4 else None,
                        "PinPoint NegRecall@10": pinpoint.get("NegRecall@10")
                        if epoch == 4
                        else None,
                        "QDiag Delta": qdiag.get("mean_target_delta") if epoch == 4 else None,
                        "QDiag Raw Win": qdiag.get("target_raw_win_rate") if epoch == 4 else None,
                        "QDiag Stable Win": qdiag.get("target_stable_win_rate")
                        if epoch == 4
                        else None,
                    }
                )
        fields = list(rows[0])
        temporary = ROOT / "all_metrics.csv.tmp"
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(ROOT / "all_metrics.csv")
        lines = [
            "# Reverse Rank: Static / Teacher / Random / No-filter / OPD",
            "",
            (
                "Original CLIP-L, S2 cache, full finetune, seed 42, four epochs, "
                "T=0.05, candidate-level KL(Student || Teacher). "
                "OPD and no_filter use epoch Top-128 -> live batch Top-32. "
                "Teacher target Top-10 filtering is computed on the selected 32; "
                "only no_filter disables it. Missing evaluations are shown as —."
            ),
            "",
        ]
        lines += ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
        for row in rows:
            lines.append(
                "| " + " | ".join("—" if row[f] is None else str(row[f]) for f in fields) + " |"
            )
        lines += ["", "## Final and best epoch", ""]
        for name in RUNS:
            available = [r for r in rows if r["Run"] == name]
            final = next(r for r in available if r["Epoch"] == 4)
            parts = []
            for metric in ("CIRR R@1", "FashionIQ R@10"):
                valid = [r for r in available if r[metric] is not None]
                if valid:
                    best = max(valid, key=lambda r: r[metric])
                    parts.append(
                        f"{metric}: E4={final[metric]}, best E{best['Epoch']}={best[metric]}"
                    )
            lines.append(f"- {name}: " + ("; ".join(parts) or "pending"))
        temporary = ROOT / "RESULTS.md.tmp"
        temporary.write_text("\n".join(lines) + "\n")
        temporary.replace(ROOT / "RESULTS.md")
    print(f"[summary] {ROOT / 'RESULTS.md'}", flush=True)


def execute(argv, log, cwd=PROJECT):
    argv = [str(x) for x in argv]
    message = "[command] " + shlex.join(argv)
    print(message, flush=True)
    log.write(message + "\n")
    log.flush()
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    with subprocess.Popen(
        argv,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, argv)


def run_one(name, extended):
    config_path = CONFIG_PATHS.get(name, PROJECT / f"configs/clip_l_reverse_rank_{name}_r128_e4.yaml")
    config = load_on_policy_config(config_path)
    run = config.experiment.output_dir
    run.mkdir(parents=True, exist_ok=True)
    with (run / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process is already running {name}") from exc
        previous = read_json(run / "effective_config.json")
        if previous:
            previous.get("mining", {}).setdefault("force_target_insertion", True)
            previous = without_loader_settings(previous)
            current = without_loader_settings(config.model_dump(mode="json"))
            for section in ("experiment", "model", "objective", "mining", "optimizer", "data"):
                if previous.get(section) != current[section]:
                    raise RuntimeError(f"Refusing to reuse {run}: {section} configuration changed")
        with (run / "runner.log").open("a", buffering=1) as log:
            py = sys.executable
            final = run / "checkpoint_epoch_004.pt"
            if not final.is_file():
                command = [
                    py,
                    PROJECT / "scripts/train_on_policy_distillation.py",
                    "--config",
                    config_path,
                ]
                if (run / "last.pt").is_file():
                    command += ["--resume-from", run / "last.pt"]
                execute(command, log)
            label = f"{LABEL_PREFIX}-{name}"
            for epoch in range(1, 5):
                checkpoint = run / f"checkpoint_epoch_{epoch:03d}.pt"
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                cirr = run / f"val_epoch_{epoch:03d}.json"
                if not cirr.is_file():
                    execute(
                        [
                            py,
                            PROJECT / "scripts/evaluate_on_policy_distillation.py",
                            "--config",
                            config_path,
                            "--checkpoint",
                            checkpoint,
                            "--output",
                            cirr,
                        ],
                        log,
                    )
                fashion = run / "fashioniq" / f"epoch_{epoch:03d}"
                if not (fashion / "metrics.json").is_file():
                    execute(
                        [
                            py,
                            PROJECT / "scripts/evaluate_fashioniq_clip.py",
                            "--model-name",
                            f"{label}-E{epoch:03d}",
                            "--config",
                            config_path,
                            "--checkpoint",
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
                            "256",
                            "--query-batch-size",
                            "256",
                            "--score-batch-size",
                            "1024",
                        ],
                        log,
                    )
            if extended:
                diag = PROJECT.parent / "diag_model"
                official = run / "pinpoint_official"
                if not (official / label / "official_metrics.csv").is_file():
                    execute(
                        [
                            py,
                            diag / "run_clip_pinpoint_official.py",
                            "--models",
                            label,
                            "--checkpoint",
                            f"{label}={final}",
                            "--output-root",
                            official,
                            "--device",
                            "cuda:0",
                            "--batch-size",
                            "128",
                        ],
                        log,
                        cwd=diag,
                    )
                qdiag = run / "qdiag"
                if not (qdiag / label / "summary.csv").is_file():
                    execute(
                        [
                            py,
                            diag / "evaluate_winrate_random10.py",
                            "--checkpoint",
                            f"{label}={final}",
                            "--output-root",
                            qdiag,
                            "--device",
                            "cuda:0",
                            "--batch-size",
                            "128",
                            "--random-samples",
                            "10",
                            "--seed",
                            "42",
                        ],
                        log,
                        cwd=diag,
                    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-indices", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--skip-extended-eval", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.summarize_only:
        summarize()
        return
    if len(set(args.run_indices)) != len(args.run_indices) or any(
        i not in range(5) for i in args.run_indices
    ):
        parser.error("run indices must be unique integers from 0 to 4")
    for index in args.run_indices:
        name = RUNS[index]
        config = load_on_policy_config(PROJECT / f"configs/clip_l_reverse_rank_{name}_r128_e4.yaml")
        assert config.objective.loss_type == "rank_kd"
        assert config.objective.rank_kl_direction == "reverse"
        assert config.experiment.initial_checkpoint is None
        assert not config.objective.anchor_kl_enabled
        print(
            f"[{index}] {name}: {config.mining.candidate_source}, "
            f"refresh={config.mining.refresh_mode}, "
            f"reservoir={config.mining.reservoir_top_k}, K={config.mining.top_k}, "
            f"Teacher filter={config.objective.teacher_target_max_rank}, "
            f"output={config.experiment.output_dir}",
            flush=True,
        )
        if not args.dry_run:
            cache = config.data.teacher_cache_dir
            if not read_json(cache / "metadata.json").get("complete"):
                raise RuntimeError(f"Missing/incomplete shared S2 cache: {cache}")
    if args.dry_run:
        return
    for index in args.run_indices:
        run_one(RUNS[index], extended=not args.skip_extended_eval)
    if not args.skip_summary:
        summarize()


if __name__ == "__main__":
    main()
