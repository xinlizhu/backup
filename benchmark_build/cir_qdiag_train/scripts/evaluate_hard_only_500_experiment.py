from __future__ import annotations

import argparse
import csv
import json
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


def _load_or_evaluate(
    *,
    name: str,
    config,
    checkpoint: Path | None,
    output_dir: Path,
    force: bool,
) -> dict:
    metrics_path = output_dir / f"{name}.json"
    if metrics_path.is_file() and not force:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        print(f"[cache] {name}: {metrics_path}", flush=True)
        return metrics
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    print(f"[evaluate] {name}: {checkpoint or 'original pretrained CLIP-L'}", flush=True)
    return evaluate_cirr(
        config,
        checkpoint_path=checkpoint,
        output_path=metrics_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare original CLIP-L, normal Full baseline, and selected hard-only "
            "checkpoints on CIRR validation."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", default="2,6,10")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    hard_config = load_config(args.config)
    baseline_config = load_config(args.baseline_config)
    epochs = [int(item.strip()) for item in args.epochs.split(",") if item.strip()]
    if not epochs or len(epochs) != len(set(epochs)):
        raise ValueError("--epochs must contain unique comma-separated integers")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selections: list[tuple[str, object, Path | None]] = [
        ("CLIP-L-original", baseline_config, None),
        ("CLIP-L-full-baseline", baseline_config, args.baseline_checkpoint.resolve()),
    ]
    selections.extend(
        (
            f"CLIP-L-hard-only-500-v8-e{epoch:03d}",
            hard_config,
            (args.run_dir / f"checkpoint_epoch_{epoch:03d}.pt").resolve(),
        )
        for epoch in epochs
    )

    rows = []
    for name, config, checkpoint in selections:
        metrics = _load_or_evaluate(
            name=name,
            config=config,
            checkpoint=checkpoint,
            output_dir=output_dir,
            force=args.force,
        )
        missing = [metric for metric in METRICS if metric not in metrics]
        if missing:
            raise ValueError(f"{name} is missing metrics: {missing}")
        rows.append(
            {
                "model": name,
                "checkpoint": metrics["checkpoint"],
                "num_queries": metrics["num_queries"],
                **{metric: metrics[metric] for metric in METRICS},
            }
        )

    summary_path = output_dir / "cirr_val_comparison.csv"
    temporary = summary_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(summary_path)
    print(summary_path.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
