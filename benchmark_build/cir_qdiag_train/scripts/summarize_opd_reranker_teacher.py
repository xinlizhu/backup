#!/usr/bin/env python3
"""Summarize the reranker-Teacher run and the S2 reverse-KL control."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_SPECS = (
    (
        "CLIP-L ← Reranker (forward KL)",
        "CLIP-L-OPD-Qwen3VL-Reranker2B",
        "clip_l_from_qwen3_vl_reranker_2b",
    ),
    (
        "CLIP-L ← Reranker (reverse KL)",
        "CLIP-L-OPD-Qwen3VL-Reranker2B-ReverseKL",
        "clip_l_from_qwen3_vl_reranker_2b_reverse_kl",
    ),
    (
        "CLIP-L ← S2 (reverse KL)",
        "CLIP-L-OPD-S2-ReverseKL",
        "clip_l_from_s2_reverse_kl",
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/opd_reranker_teacher",
    )
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    markdown = _render(root)
    destination = root / "RESULTS.md"
    destination.write_text(markdown, encoding="utf-8")
    print(destination, flush=True)


def _render(root: Path) -> str:
    mining_rows: list[list[str]] = []
    cirr_rows: list[list[str]] = []
    final_rows: list[list[str]] = []
    for label, model_name, run_name in MODEL_SPECS:
        run = root / run_name
        for epoch in range(1, 5):
            metadata = _read_json(
                run / "on_policy" / f"epoch_{epoch:03d}" / "metadata.json"
            )
            if metadata:
                mining_rows.append(
                    [
                        label,
                        str(epoch),
                        _format(metadata.get("raw_target_recall_at_k")),
                        _format(metadata.get("teacher_target_top1_rate")),
                        _format(metadata.get("teacher_target_top10_rate")),
                        _format(metadata.get("teacher_target_mean_rank")),
                        _format(metadata.get("teacher_score_mean")),
                        _format(metadata.get("teacher_score_std")),
                        _format(metadata.get("teacher_rank_entropy")),
                        _format(metadata.get("elapsed_seconds"), digits=1),
                    ]
                )
            metrics = _read_json(run / f"val_epoch_{epoch:03d}.json")
            if metrics:
                cirr_rows.append(
                    [
                        label,
                        str(epoch),
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
        final_rows.append(
            [
                label,
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
        "# Reranker Teacher 与 reverse-KL 对照实验\n\n"
        "两个 Student 都从原始 CLIP-L 独立开始全量微调，按 epoch 动态召回 Student Top-32 "
        "并强制加入 target，固定 `T=0.05`，只保留 Teacher target rank ≤ 10 的 query，"
        "训练 4 epochs、每轮保存。主实验使用 Qwen3-VL-Reranker-2B 原始 0–1 score 的 "
        "forward KL，并在完全相同协议下增加 Reranker reverse KL；另一对照使用冻结 "
        "BGE-VL-MLLM-S2 embedding score 的 reverse KL。\n\n"
        "## Candidate mining 与 Teacher\n\n"
        + _table(
            [
                "Experiment",
                "Epoch",
                "Student raw target R@32",
                "Teacher target Top-1",
                "Teacher target Top-10",
                "Teacher target mean rank",
                "Teacher score mean",
                "Teacher score std",
                "Teacher entropy",
                "Mining+scoring seconds",
            ],
            mining_rows,
        )
        + "\n\n## CIRR val checkpoint curve\n\n"
        + _table(
            [
                "Experiment",
                "Epoch",
                "R@1",
                "R@5",
                "R@10",
                "R@50",
                "Group R@1",
                "Group R@2",
                "Group R@3",
            ],
            cirr_rows,
        )
        + "\n\n## Final checkpoint\n\n"
        + _table(
            [
                "Model",
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
            final_rows,
        )
        + "\n\n`—` 表示对应阶段尚未运行；重新执行总脚本会续跑并补齐。\n"
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
