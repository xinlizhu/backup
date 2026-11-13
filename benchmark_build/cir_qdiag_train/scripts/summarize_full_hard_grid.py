from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = (
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
    "recall_at_50",
    "group_recall_at_1",
    "group_recall_at_2",
    "group_recall_at_3",
)


def _directory_name(weight: str) -> str:
    if float(weight) == 0.0:
        return "baseline"
    return f"hard_w{weight.replace('.', '_')}"


def _read_metrics(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected metric object: {path}")
    missing = [metric for metric in METRICS if metric not in value]
    if missing:
        raise ValueError(f"missing metrics in {path}: {missing}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize the CIRR validation metrics for a Full hard-weight grid."
    )
    parser.add_argument("--root", type=Path, default=Path("outputs/clip_l/full"))
    parser.add_argument("--weights", default="0.0,0.1,0.25,0.5,0.75,1.0")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/clip_l/full/hard_weight_grid_cirr.csv"),
    )
    args = parser.parse_args()

    weights = sorted(
        [item.strip() for item in args.weights.split(",") if item.strip()],
        key=float,
    )
    if not weights or len(weights) != len(set(weights)):
        raise ValueError("--weights must contain unique comma-separated values")
    if any(float(weight) < 0 for weight in weights):
        raise ValueError("hard-negative weights must be non-negative")

    payloads = []
    for weight in weights:
        path = args.root / _directory_name(weight) / "val_metrics.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payloads.append((weight, path, _read_metrics(path)))

    baseline_candidates = [item for item in payloads if float(item[0]) == 0.0]
    if len(baseline_candidates) != 1:
        raise ValueError("--weights must contain exactly one zero-weight baseline")
    baseline = baseline_candidates[0][2]
    rows = []
    for weight, path, metrics in payloads:
        row: dict[str, str | float | int] = {
            "hard_negative_weight": float(weight),
            "run": _directory_name(weight),
            "num_queries": int(metrics["num_queries"]),
            "metrics_path": str(path.resolve()),
        }
        for metric in METRICS:
            value = float(metrics[metric])
            row[metric] = value
            row[f"{metric}_minus_baseline"] = round(value - float(baseline[metric]), 4)
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(args.output)
    print(args.output.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
