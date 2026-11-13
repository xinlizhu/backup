#!/usr/bin/env python3
"""Summarize the controlled OPD K, KL direction, and temperature ablation."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class RunSpec:
    label: str
    top_k: int
    kl_direction: str
    temperature: float
    run_dir: Path
    model_name: str
    pinpoint_dir: Path
    qdiag_dir: Path
    fashioniq_dir: Path
    role: str = "new"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/opd_k_tau_kl_ablation",
    )
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    specs = _specifications(root)
    final_rows = [_final_row(spec) for spec in specs]
    _write_csv(root / "all_metrics.csv", final_rows)
    destination = root / "RESULTS.md"
    destination.write_text(_render(specs, final_rows), encoding="utf-8")
    print(destination, flush=True)


def _specifications(root: Path) -> list[RunSpec]:
    baseline = PROJECT_ROOT / "outputs/rank_candidate_policy_e4"
    return [
        RunSpec(
            "K32 Forward T=0.05 (existing baseline)",
            32,
            "forward",
            0.05,
            baseline / "dynamic_student_rank",
            "DynamicStudentRank",
            baseline / "pinpoint_official/DynamicStudentRank",
            baseline / "pinpoint_diagnostics/DynamicStudentRank",
            baseline / "fashioniq/dynamic_student_rank",
            "existing baseline",
        ),
        RunSpec(
            "K16 Forward T=0.05",
            16,
            "forward",
            0.05,
            root / "k16_forward_t005",
            "CLIP-L-OPD-S2-K16-FWD-T005",
            root / "pinpoint_official/CLIP-L-OPD-S2-K16-FWD-T005",
            root / "qdiag/CLIP-L-OPD-S2-K16-FWD-T005",
            root / "fashioniq/k16_forward_t005",
        ),
        RunSpec(
            "K8 Forward T=0.05",
            8,
            "forward",
            0.05,
            root / "k8_forward_t005",
            "CLIP-L-OPD-S2-K8-FWD-T005",
            root / "pinpoint_official/CLIP-L-OPD-S2-K8-FWD-T005",
            root / "qdiag/CLIP-L-OPD-S2-K8-FWD-T005",
            root / "fashioniq/k8_forward_t005",
        ),
        RunSpec(
            "K32 Reverse T=0.05",
            32,
            "reverse",
            0.05,
            root / "k32_reverse_t005",
            "CLIP-L-OPD-S2-K32-REV-T005",
            root / "pinpoint_official/CLIP-L-OPD-S2-K32-REV-T005",
            root / "qdiag/CLIP-L-OPD-S2-K32-REV-T005",
            root / "fashioniq/k32_reverse_t005",
        ),
        RunSpec(
            "K32 Forward T=1 (no scaling)",
            32,
            "forward",
            1.0,
            root / "k32_forward_t1",
            "CLIP-L-OPD-S2-K32-FWD-T1",
            root / "pinpoint_official/CLIP-L-OPD-S2-K32-FWD-T1",
            root / "qdiag/CLIP-L-OPD-S2-K32-FWD-T1",
            root / "fashioniq/k32_forward_t1",
        ),
        RunSpec(
            "K32 Reverse T=1 (no scaling)",
            32,
            "reverse",
            1.0,
            root / "k32_reverse_t1",
            "CLIP-L-OPD-S2-K32-REV-T1",
            root / "pinpoint_official/CLIP-L-OPD-S2-K32-REV-T1",
            root / "qdiag/CLIP-L-OPD-S2-K32-REV-T1",
            root / "fashioniq/k32_reverse_t1",
        ),
    ]


def _final_row(spec: RunSpec) -> dict[str, Any]:
    cirr = _read_json(spec.run_dir / "val_epoch_004.json")
    pinpoint = _read_csv_first(spec.pinpoint_dir / "official_metrics.csv")
    qdiag = _read_csv_subset(spec.qdiag_dir / "summary.csv", "strict")
    fashioniq = _read_json(spec.fashioniq_dir / "metrics.json").get(
        "macro_average", {}
    )
    timing = _read_json(spec.run_dir / "training_time.json")
    total_seconds = sum(
        float(row.get("total_seconds", 0.0))
        for row in timing.get("epochs_this_process", [])
    )
    return {
        "Model": spec.label,
        "Role": spec.role,
        "K": spec.top_k,
        "KL": spec.kl_direction,
        "Temperature": spec.temperature,
        "Training hours": total_seconds / 3600.0 if total_seconds else None,
        "CIRR R@1": cirr.get("recall_at_1"),
        "CIRR R@5": cirr.get("recall_at_5"),
        "CIRR R@10": cirr.get("recall_at_10"),
        "CIRR R@50": cirr.get("recall_at_50"),
        "Group R@1": cirr.get("group_recall_at_1"),
        "Group R@2": cirr.get("group_recall_at_2"),
        "Group R@3": cirr.get("group_recall_at_3"),
        "PinPoint mAP@10": pinpoint.get("mAP@10"),
        "PinPoint NegRecall@10": pinpoint.get("NegRecall@10"),
        "FashionIQ R@10": fashioniq.get("recall_at_10"),
        "FashionIQ R@50": fashioniq.get("recall_at_50"),
        "QDiag Delta": qdiag.get("mean_target_delta"),
        "QDiag Raw Win": qdiag.get("target_raw_win_rate"),
        "QDiag Stable Win": qdiag.get("target_stable_win_rate"),
        "QDiag tau95": qdiag.get("normal_perturbation_threshold"),
        "QDiag Own-Random": qdiag.get(
            "positive_query_own_minus_random_target"
        ),
    }


def _render(specs: list[RunSpec], final_rows: list[dict[str, Any]]) -> str:
    curve_rows: list[list[str]] = []
    mining_rows: list[list[str]] = []
    for spec in specs:
        for epoch in range(1, 5):
            cirr = _read_json(spec.run_dir / f"val_epoch_{epoch:03d}.json")
            if cirr:
                curve_rows.append(
                    [
                        spec.label,
                        str(epoch),
                        _format(cirr.get("recall_at_1")),
                        _format(cirr.get("recall_at_5")),
                        _format(cirr.get("recall_at_10")),
                        _format(cirr.get("recall_at_50")),
                        _format(cirr.get("group_recall_at_1")),
                    ]
                )
        first = _read_json(spec.run_dir / "on_policy/epoch_001/metadata.json")
        last = _read_json(spec.run_dir / "on_policy/epoch_004/metadata.json")
        if first:
            mining_rows.append(
                [
                    spec.label,
                    _format(first.get("raw_target_recall_at_k")),
                    _format(last.get("raw_target_recall_at_k")),
                    _format(first.get("teacher_target_top10_rate")),
                    _format(last.get("teacher_target_top10_rate")),
                    _format(first.get("teacher_rank_entropy")),
                    _format(last.get("teacher_rank_entropy")),
                ]
            )
    final_headers = [
        "Model",
        "Training hours",
        "CIRR R@1",
        "CIRR R@10",
        "Group R@1",
        "PinPoint mAP@10",
        "PinPoint NegRecall@10",
        "FashionIQ R@10",
        "FashionIQ R@50",
        "QDiag Delta",
        "QDiag Raw Win",
        "QDiag Stable Win",
        "QDiag tau95",
        "QDiag Own-Random",
    ]
    final_table_rows = [
        [_format(row.get(header)) for header in final_headers]
        for row in final_rows
    ]
    return (
        "# OPD candidate K / KL direction / temperature ablation\n\n"
        "所有新实验均从同一个原始 CLIP-L 初始化，使用冻结的 BGE-VL-MLLM-S2 "
        "Teacher、Student 动态 Top-K、target 强制注入、Teacher target Top-10 筛选、"
        "full finetune、seed 42，并训练 4 epochs、每轮保存。除表中 K、KL 方向和 "
        "temperature 外，其余配置完全相同。\n\n"
        "Forward 为 `D_KL(p_teacher || p_student)`，Reverse 为 "
        "`D_KL(p_student || p_teacher)`。`T=1` 表示不额外缩放 cosine similarity，"
        "即直接计算 `softmax(similarity)`；并不是取消 softmax。已有 K32/Forward/"
        "T=0.05 checkpoint 作为统一 baseline，不重复训练。\n\n"
        "## CIRR checkpoint curve\n\n"
        + _table(
            ["Model", "Epoch", "R@1", "R@5", "R@10", "R@50", "Group R@1"],
            curve_rows,
        )
        + "\n\n## Candidate mining diagnostics\n\n"
        + _table(
            [
                "Model",
                "E1 raw target R@K",
                "E4 raw target R@K",
                "E1 Teacher target Top-10",
                "E4 Teacher target Top-10",
                "E1 Teacher entropy",
                "E4 Teacher entropy",
            ],
            mining_rows,
        )
        + "\n\n## Final checkpoint: all datasets\n\n"
        + _table(final_headers, final_table_rows)
        + "\n\n`—` 表示对应训练或评估尚未完成；重新运行总脚本会从已有 checkpoint "
        "续跑并补齐结果。\n"
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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _format(value: Any, *, digits: int = 4) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return value
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
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
