#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


VARIANTS = (
    ("dynamic_student_rank", "Dynamic Student Top-K + Rank KD"),
    ("teacher_topk_rank", "Teacher Top-K + Rank KD"),
    ("random_rank", "Random + Rank KD"),
    ("static_student_rank", "Static Student Top-K + Rank KD"),
    ("dynamic_student_infonce", "Dynamic Student Top-K + InfoNCE"),
    (
        "dynamic_student_infonce_teacher_top10",
        "Dynamic Student Top-K + Teacher-Top10-filtered InfoNCE",
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    lines = [
        "# Rank candidate-policy ablation",
        "",
        "所有模型从同一原始 CLIP-L 初始化，全量微调 4 epochs，K=32，"
        "每轮保存。Rank KD 使用 Teacher target Top-10 置信筛选；"
        "InfoNCE 使用 target 对当前 31 个 Student hard negatives 的局部交叉熵，"
        "并分别比较不筛选和 Teacher target Top-10 筛选。",
        "",
        "| Run | Candidate source | Refresh | Objective |",
        "|---|---|---|---|",
        "| Dynamic Student Top-K + Rank KD | Student | every epoch | Teacher→Student KL |",
        "| Teacher Top-K + Rank KD | Teacher | once | Teacher→Student KL |",
        "| Random + Rank KD | uniform random non-targets | every epoch | Teacher→Student KL |",
        "| Static Student Top-K + Rank KD | initial Student | once | Teacher→Student KL |",
        "| Dynamic Student Top-K + InfoNCE | Student | every epoch | target cross-entropy |",
        "| Dynamic Student Top-K + Teacher-Top10-filtered InfoNCE | Student | every epoch | target cross-entropy, Teacher target rank ≤ 10 only |",
        "",
    ]
    _append_cirr(lines, root)
    _append_mining(lines, root)
    _append_pinpoint(lines, root)
    _append_qdiag(lines, root)
    _append_fashioniq(lines, root)
    output = root / "RESULTS.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    temporary.replace(output)


def _append_cirr(lines: list[str], root: Path) -> None:
    rows: list[tuple[str, int, dict[str, Any]]] = []
    for run_name, label in VARIANTS:
        for epoch in range(1, 5):
            path = root / run_name / f"val_epoch_{epoch:03d}.json"
            if path.is_file():
                rows.append((label, epoch, _read_json(path)))
    if not rows:
        return
    lines.extend(
        [
            "## CIRR validation by epoch",
            "",
            "| Run | Epoch | R@1 | R@5 | R@10 | R@50 | Group R@1 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, epoch, row in rows:
        lines.append(
            f"| {label} | {epoch} | {_f(row, 'recall_at_1')} | "
            f"{_f(row, 'recall_at_5')} | {_f(row, 'recall_at_10')} | "
            f"{_f(row, 'recall_at_50')} | {_f(row, 'group_recall_at_1')} |"
        )
    lines.append("")


def _append_mining(lines: list[str], root: Path) -> None:
    rows: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for run_name, label in VARIANTS:
        first_path = root / run_name / "on_policy/epoch_001/metadata.json"
        last_path = root / run_name / "on_policy/epoch_004/metadata.json"
        if first_path.is_file():
            first = _read_json(first_path)
            last = _read_json(last_path) if last_path.is_file() else first
            rows.append((label, first, last))
    if not rows:
        return
    lines.extend(
        [
            "## Candidate mining diagnostics",
            "",
            "`source target R@32` 表示 target 在强制注入前是否已出现在候选来源的原始 Top-32；"
            "Random 明确排除 target，因此该值按定义为 0。",
            "",
            "| Run | Source | Refresh | Epoch-1 source target R@32 | Final source target R@32 | Epoch-1 Teacher target Top-10 | Final Teacher target Top-10 |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for label, first, last in rows:
        lines.append(
            f"| {label} | {first.get('candidate_source', '—')} | "
            f"{first.get('refresh_mode', '—')} | "
            f"{_pct(first.get('raw_target_recall_at_k'))} | "
            f"{_pct(last.get('raw_target_recall_at_k'))} | "
            f"{_pct(first.get('teacher_target_top10_rate'))} | "
            f"{_pct(last.get('teacher_target_top10_rate'))} |"
        )
    lines.append("")


def _append_pinpoint(lines: list[str], root: Path) -> None:
    path = root / "pinpoint_official/official_map10_negrecall10.csv"
    if not path.is_file():
        return
    rows = _read_csv(path)
    lines.extend(
        [
            "## PinPoint official",
            "",
            "| Model | mAP@10 ↑ | NegRecall@10 ↓ | Queries |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['mAP@10']} | {row['NegRecall@10']} | "
            f"{row['total_queries']} |"
        )
    lines.append("")


def _append_qdiag(lines: list[str], root: Path) -> None:
    path = root / "pinpoint_diagnostics/overview_all_strict.csv"
    if not path.is_file():
        return
    rows = [row for row in _read_csv(path) if row.get("subset") == "strict"]
    lines.extend(
        [
            "## QDiag strict ADD/SUBSTITUTE",
            "",
            "| Model | Negatives | tau95 | Mean Delta | Raw Win | Stable Win |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['negative_count']} | "
            f"{_float_text(row['normal_perturbation_threshold'])} | "
            f"{_float_text(row['mean_target_delta'])} | "
            f"{_float_text(row['target_raw_win_rate'])} | "
            f"{_float_text(row['target_stable_win_rate'])} |"
        )
    lines.append("")


def _append_fashioniq(lines: list[str], root: Path) -> None:
    rows: list[tuple[str, dict[str, Any]]] = []
    for run_name, label in VARIANTS:
        path = root / "fashioniq" / run_name / "metrics.json"
        if path.is_file():
            rows.append((label, _read_json(path)))
    if not rows:
        return
    lines.extend(
        [
            "## FashionIQ validation transfer",
            "",
            "| Run | Average R@10 | Average R@50 | Queries |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, row in rows:
        macro = row["macro_average"]
        lines.append(
            f"| {label} | {_f(macro, 'recall_at_10')} | "
            f"{_f(macro, 'recall_at_50')} | {row['num_queries']} |"
        )
    lines.append("")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _f(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    return "—" if value is None else f"{float(value):.4f}"


def _pct(value: Any) -> str:
    return "—" if value is None else f"{100.0 * float(value):.2f}%"


def _float_text(value: Any) -> str:
    return "—" if value in (None, "") else f"{float(value):.6f}"


if __name__ == "__main__":
    main()
