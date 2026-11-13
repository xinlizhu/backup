from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _read_training_seconds(path: Path) -> dict[str, int]:
    totals: dict[str, int] = {}
    if not path.is_file():
        return totals
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["phase"] != "train" or row["status"] not in {"ok", "resumed"}:
                continue
            totals[row["run"]] = totals.get(row["run"], 0) + int(row["seconds"])
    return totals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    training_seconds = _read_training_seconds(root / "timing.tsv")
    pinpoint_path = root / "pinpoint_official" / "official_map10_negrecall10.csv"
    pinpoint: dict[str, dict[str, str]] = {}
    if pinpoint_path.is_file():
        with pinpoint_path.open(encoding="utf-8", newline="") as handle:
            pinpoint = {row["model"]: row for row in csv.DictReader(handle)}

    final_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in root.glob("m*_w*") if path.is_dir()):
        config = _read_json(run_dir / "effective_config.json")
        optimizer = config["optimizer"]
        margin = float(optimizer["hard_negative_margin"])
        weight = float(optimizer["hard_negative_weight"])
        run = run_dir.name
        for epoch in (20, 40, 60):
            metrics_path = run_dir / f"val_metrics_epoch_{epoch:03d}.json"
            if not metrics_path.is_file():
                continue
            metrics = _read_json(metrics_path)
            curve_rows.append(
                {
                    "run": run,
                    "margin": margin,
                    "hard_weight": weight,
                    "epoch": epoch,
                    **{
                        key: metrics[key]
                        for key in metrics
                        if key.startswith(("recall_at_", "group_recall_at_"))
                    },
                }
            )
        final_metrics = _read_json(run_dir / "val_metrics_epoch_060.json")
        reverse = _read_json(run_dir / "reverse_val_metrics.json")
        model_name = f"CLIP-L-margin-{run}-e60"
        official = pinpoint.get(model_name, {})
        final_rows.append(
            {
                "run": run,
                "margin": margin,
                "hard_weight": weight,
                "training_seconds": training_seconds.get(run, ""),
                "training_hours": (
                    round(training_seconds[run] / 3600, 4)
                    if run in training_seconds
                    else ""
                ),
                **{
                    key: final_metrics[key]
                    for key in final_metrics
                    if key.startswith(("recall_at_", "group_recall_at_"))
                },
                "reverse_mrr": reverse["mrr"],
                "reverse_map_at_10": reverse["map_at_10"],
                "reverse_ndcg_at_10": reverse["ndcg_at_10"],
                "pinpoint_map_at_10": official.get("mAP@10", ""),
                "pinpoint_neg_recall_at_10": official.get("NegRecall@10", ""),
            }
        )

    for path, rows in (
        (root / "grid_summary.csv", final_rows),
        (root / "cirr_checkpoint_curve.csv", curve_rows),
    ):
        if not rows:
            continue
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(path)


if __name__ == "__main__":
    main()
