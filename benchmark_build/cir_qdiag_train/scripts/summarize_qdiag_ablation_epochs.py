#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


VARIANTS = (
    ("DynamicStudentRank", "Dynamic Student Rank"),
    ("TeacherTopKRank", "Teacher Top-K Rank"),
    ("RandomRank", "Random Rank"),
    ("StaticStudentRank", "Static Student Rank"),
    ("DynamicStudentInfoNCE", "Dynamic Student InfoNCE"),
    (
        "DynamicStudentInfoNCETeacherTop10",
        "Dynamic Student InfoNCE + Teacher Top-10",
    ),
)

SUBSETS = (
    "strict",
    "add",
    "substitute",
    "delete",
    "strict|object",
    "strict|attribute",
    "strict|predicate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    return parser.parse_args()


def evaluation_name(base_name: str, epoch: int) -> str:
    return base_name if epoch == 4 else f"{base_name}_E{epoch:02d}"


def main() -> None:
    args = parse_args()
    rows = _read_csv(args.input.resolve())
    indexed = {(row["model"], row["subset"]): row for row in rows}
    missing = [
        (evaluation_name(base_name, epoch), subset)
        for base_name, _ in VARIANTS
        for epoch in range(1, 5)
        for subset in SUBSETS
        if (evaluation_name(base_name, epoch), subset) not in indexed
    ]
    if missing:
        preview = ", ".join(f"{model}:{subset}" for model, subset in missing[:8])
        raise ValueError(f"QDiag summary is missing {len(missing)} rows: {preview}")

    flat_rows = _build_flat_rows(indexed)
    _write_csv(args.csv_output.resolve(), flat_rows)
    lines = _build_markdown(indexed, flat_rows, args.input.resolve())
    _write_text(args.output.resolve(), "\n".join(lines).rstrip() + "\n")


def _build_flat_rows(
    indexed: dict[tuple[str, str], dict[str, str]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for base_name, label in VARIANTS:
        for epoch in range(1, 5):
            model_name = evaluation_name(base_name, epoch)
            strict = indexed[(model_name, "strict")]
            result.append(
                {
                    "model": label,
                    "epoch": epoch,
                    "tau95": _number(strict, "normal_perturbation_threshold"),
                    "strict_mean_delta": _number(strict, "mean_target_delta"),
                    "strict_raw_win": _number(strict, "target_raw_win_rate"),
                    "strict_stable_win": _number(strict, "target_stable_win_rate"),
                    "own_target_minus_random_target": _number(
                        strict, "positive_query_own_minus_random_target"
                    ),
                    "add_stable_win": _stable(indexed, model_name, "add"),
                    "substitute_stable_win": _stable(
                        indexed, model_name, "substitute"
                    ),
                    "delete_stable_win": _stable(indexed, model_name, "delete"),
                    "object_stable_win": _stable(
                        indexed, model_name, "strict|object"
                    ),
                    "attribute_stable_win": _stable(
                        indexed, model_name, "strict|attribute"
                    ),
                    "predicate_stable_win": _stable(
                        indexed, model_name, "strict|predicate"
                    ),
                }
            )
    return result


def _build_markdown(
    indexed: dict[tuple[str, str], dict[str, str]],
    flat_rows: list[dict[str, Any]],
    input_path: Path,
) -> list[str]:
    lines = [
        "# QDiag: six ablations × four epochs",
        "",
        "每个 checkpoint 均在同一 CIR-QDiag 上评估。`Delta = sim(q+, target) - "
        "sim(q−, target)`；Raw Win 使用 `Delta > 0`，Stable Win 使用各 checkpoint "
        "独立校准的 `Delta > tau95`。strict 只包含 ADD/SUBSTITUTE。",
        "",
        "## Strict results by epoch",
        "",
        "| Model | Epoch | tau95 | Mean Delta | Raw Win | Stable Win | Own target - random target |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in flat_rows:
        lines.append(
            f"| {row['model']} | {row['epoch']} | {_f(row['tau95'])} | "
            f"{_f(row['strict_mean_delta'])} | {_f(row['strict_raw_win'])} | "
            f"{_f(row['strict_stable_win'])} | "
            f"{_f(row['own_target_minus_random_target'])} |"
        )

    lines.extend(
        [
            "",
            "## Best epoch by strict Stable Win",
            "",
            "| Model | Best epoch | Stable Win | Mean Delta | Raw Win | tau95 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, label in VARIANTS:
        candidates = [row for row in flat_rows if row["model"] == label]
        best = max(candidates, key=lambda row: row["strict_stable_win"])
        lines.append(
            f"| {label} | {best['epoch']} | {_f(best['strict_stable_win'])} | "
            f"{_f(best['strict_mean_delta'])} | {_f(best['strict_raw_win'])} | "
            f"{_f(best['tau95'])} |"
        )

    lines.extend(
        [
            "",
            "## Operation Stable Win by epoch",
            "",
            "| Model | Epoch | ADD | SUBSTITUTE | DELETE |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in flat_rows:
        lines.append(
            f"| {row['model']} | {row['epoch']} | {_f(row['add_stable_win'])} | "
            f"{_f(row['substitute_stable_win'])} | "
            f"{_f(row['delete_stable_win'])} |"
        )

    lines.extend(
        [
            "",
            "## Semantic Stable Win by epoch",
            "",
            "| Model | Epoch | Object | Attribute | Predicate |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in flat_rows:
        lines.append(
            f"| {row['model']} | {row['epoch']} | "
            f"{_f(row['object_stable_win'])} | {_f(row['attribute_stable_win'])} | "
            f"{_f(row['predicate_stable_win'])} |"
        )

    lines.extend(
        [
            "",
            "## Source files",
            "",
            f"- Combined QDiag rows: `{input_path}`",
            "- Machine-readable per-checkpoint table: `qdiag_all_epochs_summary.csv`",
        ]
    )
    return lines


def _stable(
    indexed: dict[tuple[str, str], dict[str, str]], model: str, subset: str
) -> float:
    return _number(indexed[(model, subset)], "target_stable_win_rate")


def _number(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _f(value: Any) -> str:
    return f"{float(value):.6f}"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    main()
