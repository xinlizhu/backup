#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


RUNS = {
    "CLIP-L-KD-baseline": ("Baseline", "baseline/val_metrics.json"),
    "CLIP-L-KD-rank": ("Positive Rank KD", "positive_rank/val_epoch_016.json"),
    "CLIP-L-KD-response": (
        "Semantic Response KD",
        "semantic_response/val_epoch_016.json",
    ),
    "CLIP-L-KD-joint": ("Rank + Response KD", "joint/val_epoch_016.json"),
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(value: object, digits: int = 4) -> str:
    if value in (None, ""):
        return "—"
    return f"{float(value):.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    official = {
        row["model"]: row
        for row in _read_csv(root / "pinpoint_official/official_map10_negrecall10.csv")
    }
    diagnostic_rows = _read_csv(
        root / "pinpoint_diagnostics/combined_summary.csv"
    )
    diagnostics = {
        (row["model"], row["subset"]): row
        for row in diagnostic_rows
    }
    combined: list[dict[str, object]] = []
    for model_name, (label, relative_metrics) in RUNS.items():
        cirr_path = root / relative_metrics
        cirr = (
            json.loads(cirr_path.read_text(encoding="utf-8"))
            if cirr_path.is_file()
            else {}
        )
        row = {
            "model": model_name,
            "experiment": label,
            **{key: cirr.get(key) for key in (
                "recall_at_1",
                "recall_at_5",
                "recall_at_10",
                "recall_at_50",
                "group_recall_at_1",
                "group_recall_at_2",
                "group_recall_at_3",
            )},
            "mAP@10": official.get(model_name, {}).get("mAP@10"),
            "NegRecall@10": official.get(model_name, {}).get("NegRecall@10"),
            "strict_mean_delta": diagnostics.get((model_name, "strict"), {}).get(
                "mean_target_delta"
            ),
            "strict_raw_win": diagnostics.get((model_name, "strict"), {}).get(
                "target_raw_win_rate"
            ),
            "strict_stable_win": diagnostics.get((model_name, "strict"), {}).get(
                "target_stable_win_rate"
            ),
            "object_mean_delta": diagnostics.get(
                (model_name, "strict|object"), {}
            ).get("mean_target_delta"),
            "object_raw_win": diagnostics.get(
                (model_name, "strict|object"), {}
            ).get("target_raw_win_rate"),
            "object_stable_win": diagnostics.get(
                (model_name, "strict|object"), {}
            ).get("target_stable_win_rate"),
            "stable_threshold": diagnostics.get((model_name, "strict"), {}).get(
                "normal_perturbation_threshold"
            ),
            "positive_own_minus_random": diagnostics.get(
                (model_name, "strict"), {}
            ).get(
                "positive_query_own_minus_random_target"
            ),
        }
        combined.append(row)

    csv_path = root / "ablation_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)

    lines = [
        "# Semantic Ability Distillation Ablation",
        "",
        "All trained variants start from the same CLIP-L Full baseline. The vision "
        "encoder is frozen; no InfoNCE loss is used.",
        "",
        "| Experiment | CIRR R@1 | R@5 | R@10 | Group R@1 | mAP@10 | NegRecall@10 | Object Delta | Object Raw Win | Object Stable Win | Strict Stable Win | τ95 | Own−Random |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in combined:
        lines.append(
            "| {experiment} | {r1} | {r5} | {r10} | {gr1} | {map10} | {neg10} | "
            "{object_delta} | {object_raw} | {object_stable} | {strict_stable} | "
            "{threshold} | {random_gap} |".format(
                experiment=row["experiment"],
                r1=_number(row["recall_at_1"]),
                r5=_number(row["recall_at_5"]),
                r10=_number(row["recall_at_10"]),
                gr1=_number(row["group_recall_at_1"]),
                map10=_number(row["mAP@10"], 6),
                neg10=_number(row["NegRecall@10"], 6),
                object_delta=_number(row["object_mean_delta"], 6),
                object_raw=_number(row["object_raw_win"], 6),
                object_stable=_number(row["object_stable_win"], 6),
                strict_stable=_number(row["strict_stable_win"], 6),
                threshold=_number(row["stable_threshold"], 6),
                random_gap=_number(row["positive_own_minus_random"], 6),
            )
        )
    (root / "ABLATION_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
