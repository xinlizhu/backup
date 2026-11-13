from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from cir_qdiag_train.config import load_config
from cir_qdiag_train.evaluation import evaluate_cirr


METRICS = (
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
    "recall_at_50",
    "group_recall_at_1",
    "group_recall_at_2",
    "group_recall_at_3",
)
CHECKPOINT_PATTERN = re.compile(r"checkpoint_epoch_(\d+)\.pt$")


def _discover_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    checkpoints = []
    for path in run_dir.glob("checkpoint_epoch_*.pt"):
        match = CHECKPOINT_PATTERN.fullmatch(path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
    return sorted(checkpoints)


def _read_metrics(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected metric object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate every saved epoch checkpoint on CIRR validation."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Defaults to experiment.output_dir from the config.",
    )
    parser.add_argument("--epochs", default=None, help="Optional comma-separated epochs.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = (args.run_dir or config.experiment.output_dir).resolve()
    checkpoints = _discover_checkpoints(run_dir)
    if args.epochs:
        selected = {int(value.strip()) for value in args.epochs.split(",") if value.strip()}
        checkpoints = [item for item in checkpoints if item[0] in selected]
        missing = sorted(selected - {epoch for epoch, _ in checkpoints})
        if missing:
            raise FileNotFoundError(f"missing requested checkpoint epochs: {missing}")
    if not checkpoints:
        raise FileNotFoundError(f"no epoch checkpoints found in {run_dir}")

    output_dir = run_dir / "checkpoint_curve_cirr"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for epoch, checkpoint in checkpoints:
        metrics_path = output_dir / f"epoch_{epoch:03d}.json"
        if metrics_path.is_file() and not args.force:
            print(f"[cache] {metrics_path}", flush=True)
            metrics = _read_metrics(metrics_path)
        else:
            print(f"[evaluate] epoch={epoch} checkpoint={checkpoint}", flush=True)
            metrics = evaluate_cirr(
                config,
                checkpoint_path=checkpoint,
                output_path=metrics_path,
            )
        missing_metrics = [metric for metric in METRICS if metric not in metrics]
        if missing_metrics:
            raise ValueError(f"{metrics_path} is missing metrics: {missing_metrics}")
        rows.append(
            {
                "epoch": epoch,
                "checkpoint": str(checkpoint.resolve()),
                "num_queries": int(metrics["num_queries"]),
                **{metric: float(metrics[metric]) for metric in METRICS},
            }
        )

    summary_path = output_dir / "curve.csv"
    temporary = summary_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(summary_path)

    best = {
        metric: {
            "epoch": max(rows, key=lambda row: row[metric])["epoch"],
            "value": max(row[metric] for row in rows),
        }
        for metric in METRICS
    }
    best_path = output_dir / "best_by_metric.json"
    best_path.write_text(
        json.dumps(best, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary_path.read_text(encoding="utf-8"), end="")
    print(json.dumps(best, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
