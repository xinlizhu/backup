#!/usr/bin/env python3
"""Summarize S2 -> CLIP-L OPD with composed q- candidates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_SPECS = (
    (16, "CLIP-L-OPD-S2-QminusCandidates-K16", "clip_l_from_s2_mixed_qminus_top16"),
    (32, "CLIP-L-OPD-S2-QminusCandidates-K32", "clip_l_from_s2_mixed_qminus_top32"),
    (64, "CLIP-L-OPD-S2-QminusCandidates-K64", "clip_l_from_s2_mixed_qminus_top64"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/opd_s2_qminus_candidates",
    )
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "RESULTS.md"
    destination.write_text(_render(root), encoding="utf-8")
    print(destination, flush=True)


def _render(root: Path) -> str:
    rows: list[list[str]] = []
    final: list[list[str]] = []
    for top_k, model_name, run_name in MODEL_SPECS:
        run = root / run_name
        for epoch in range(1, 5):
            metadata = _read_json(
                run / "on_policy" / f"epoch_{epoch:03d}" / "metadata.json"
            )
            metrics = _read_json(run / f"val_epoch_{epoch:03d}.json")
            if metadata or metrics:
                rows.append(
                    [
                        str(top_k),
                        str(epoch),
                        _format(metadata.get("raw_target_recall_at_k")),
                        _format(
                            metadata.get("mean_negative_query_candidates_per_sample")
                        ),
                        _format(
                            metadata.get("truncated_negative_query_candidate_count"),
                            digits=0,
                        ),
                        _format(metadata.get("teacher_target_top10_rate")),
                        _format(metrics.get("recall_at_1")),
                        _format(metrics.get("recall_at_5")),
                        _format(metrics.get("recall_at_10")),
                        _format(metrics.get("recall_at_50")),
                        _format(metrics.get("group_recall_at_1")),
                        _format(metrics.get("group_recall_at_2")),
                        _format(metrics.get("group_recall_at_3")),
                    ]
                )
        pinpoint = _read_csv_first(
            root / "pinpoint_official" / model_name / "official_metrics.csv"
        )
        qdiag = _read_csv_subset(
            root / "qdiag" / model_name / "summary.csv", "strict"
        )
        fashion = _read_json(root / "fashioniq" / run_name / "metrics.json").get(
            "macro_average", {}
        )
        final.append(
            [
                str(top_k),
                _format(pinpoint.get("mAP@10")),
                _format(pinpoint.get("NegRecall@10")),
                _format(fashion.get("recall_at_10")),
                _format(fashion.get("recall_at_50")),
                _format(qdiag.get("mean_target_delta")),
                _format(qdiag.get("target_raw_win_rate")),
                _format(qdiag.get("target_stable_win_rate")),
                _format(qdiag.get("normal_perturbation_threshold")),
                _format(qdiag.get("positive_query_own_minus_random_target")),
            ]
        )
    return (
        "# S2 → CLIP-L OPD composed q− candidate K sweep\n\n"
        "每条训练样本的固定宽度候选集为：target image + 该样本 composed q− "
        "+ Student 动态检索的 hard images。q− 由 reference image 与 negative text "
        "现场组合；Teacher 和 Student 都计算 q+ 对混合候选的相似度分布。使用 S2 "
        "Teacher、forward KL、T=0.05、Teacher target Top-10 筛选、CLIP-L 全量微调，"
        "分别训练 K=16/32/64，各4 epochs。K16 每条最多插入14条 q−，以保留至少一张 "
        "hard image；K32和K64插入全部 q−。\n\n"
        "## CIRR checkpoint curve\n\n"
        + _table(
            [
                "K",
                "Epoch",
                "Raw image target R@K",
                "Mean q− candidates",
                "Truncated q−",
                "Teacher target Top-10",
                "R@1",
                "R@5",
                "R@10",
                "R@50",
                "Group R@1",
                "Group R@2",
                "Group R@3",
            ],
            rows,
        )
        + "\n\n## Final checkpoint\n\n"
        + _table(
            [
                "K",
                "PinPoint mAP@10",
                "PinPoint NegRecall@10",
                "FashionIQ R@10",
                "FashionIQ R@50",
                "QDiag Delta",
                "QDiag Raw Win",
                "QDiag Stable Win",
                "QDiag tau95",
                "QDiag Own-Random",
            ],
            final,
        )
        + "\n\n`—` 表示对应阶段尚未运行。\n"
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_csv_first(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return next(csv.DictReader(handle), {})


def _read_csv_subset(path: Path, subset: str) -> dict[str, str]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("subset") == subset:
                return row
    return {}


def _format(value: Any, *, digits: int = 4) -> str:
    if value is None or value == "":
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _table(headers: list[str], rows: list[list[str]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    divider = "|" + "|".join("---" for _ in headers) + "|"
    body = "\n".join("| " + " | ".join(row) + " |" for row in rows)
    if not body:
        body = "| " + " | ".join("—" for _ in headers) + " |"
    return "\n".join((header, divider, body))


if __name__ == "__main__":
    main()
