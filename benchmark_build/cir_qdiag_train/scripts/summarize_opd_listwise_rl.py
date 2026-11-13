#!/usr/bin/env python3
"""Summarize OPD E4, Target CE, and group-relative policy post-training."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ResultSpec:
    label: str
    run_dir: Path
    final_epoch: int
    model_name: str
    pinpoint_dir: Path
    qdiag_dir: Path
    fashioniq_dir: Path
    objective: str


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/opd_listwise_rl_e4",
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


def _specifications(root: Path) -> list[ResultSpec]:
    baseline = PROJECT_ROOT / "outputs/rank_candidate_policy_e4"
    return [
        ResultSpec(
            "OPD E4 baseline",
            baseline / "dynamic_student_rank",
            4,
            "DynamicStudentRank",
            baseline / "pinpoint_official/DynamicStudentRank",
            baseline / "pinpoint_diagnostics/DynamicStudentRank",
            baseline / "fashioniq/dynamic_student_rank",
            "Forward-KL OPD",
        ),
        ResultSpec(
            "OPD E4 → Target CE E4",
            root / "target_ce",
            4,
            "CLIP-L-OPD-E4-TargetCE",
            root / "pinpoint_official/CLIP-L-OPD-E4-TargetCE",
            root / "qdiag/CLIP-L-OPD-E4-TargetCE",
            root / "fashioniq/target_ce",
            "target one-hot CE",
        ),
        ResultSpec(
            "OPD E4 → Group-relative policy E4",
            root / "group_relative",
            4,
            "CLIP-L-OPD-E4-GroupRelative",
            root / "pinpoint_official/CLIP-L-OPD-E4-GroupRelative",
            root / "qdiag/CLIP-L-OPD-E4-GroupRelative",
            root / "fashioniq/group_relative",
            "16-rollout group-normalized target NDCG@10 policy gradient",
        ),
        ResultSpec(
            "Original CLIP-L → Group-relative policy E4",
            root / "raw_clip_group_relative",
            4,
            "CLIP-L-Raw-GroupRelative",
            root / "pinpoint_official/CLIP-L-Raw-GroupRelative",
            root / "qdiag/CLIP-L-Raw-GroupRelative",
            root / "fashioniq/raw_clip_group_relative",
            "16-rollout group-normalized target NDCG@10 policy gradient",
        ),
    ]


def _final_row(spec: ResultSpec) -> dict[str, Any]:
    cirr = _read_json(spec.run_dir / f"val_epoch_{spec.final_epoch:03d}.json")
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
        "Objective": spec.objective,
        "Additional training hours": (
            total_seconds / 3600.0 if total_seconds else None
        ),
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


def _render(specs: list[ResultSpec], final_rows: list[dict[str, Any]]) -> str:
    curve_rows: list[list[str]] = []
    for spec in specs:
        if spec.label == "OPD E4 baseline":
            epochs = (4,)
            stages = ("initial OPD",)
        else:
            epochs = (1, 2, 3, 4)
            stages = (
                "post epoch 1",
                "post epoch 2",
                "post epoch 3",
                "post epoch 4",
            )
        for epoch, stage in zip(epochs, stages, strict=True):
            metrics = _read_json(spec.run_dir / f"val_epoch_{epoch:03d}.json")
            if metrics:
                curve_rows.append(
                    [
                        spec.label,
                        stage,
                        _format(metrics.get("recall_at_1")),
                        _format(metrics.get("recall_at_5")),
                        _format(metrics.get("recall_at_10")),
                        _format(metrics.get("recall_at_50")),
                        _format(metrics.get("group_recall_at_1")),
                    ]
                )

    policy_rows = []
    for policy_spec in (
        spec for spec in specs if "Group-relative" in spec.label
    ):
        policy_logs = _last_training_record_by_epoch(
            policy_spec.run_dir / "train_metrics.jsonl"
        )
        for epoch in sorted(policy_logs):
            row = policy_logs[epoch]
            policy_rows.append(
                [
                    policy_spec.label,
                    str(epoch),
                    _format(row.get("rollout_reward_mean")),
                    _format(row.get("rollout_reward_std")),
                    _format(row.get("effective_rl_query_rate")),
                    _format(row.get("unique_rollout_count")),
                    _format(row.get("unique_rollout_rate")),
                    _format(row.get("sampled_target_recall_at_k")),
                    _format(row.get("sampled_target_rank_with_miss")),
                    _format(row.get("policy_target_probability")),
                    _format(row.get("policy_entropy")),
                ]
            )

    final_headers = [
        "Model",
        "Additional training hours",
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
        "# OPD → Group-relative policy retrieval post-training\n\n"
        "实验比较 OPD E4 → Target CE、OPD E4 → group-relative policy，以及"
        "原始 CLIP-L → group-relative policy。三者均使用 Student 动态 Top-32、"
        "target 强制加入候选池、全量微调、相同 seed/batch size/学习率，并训练 "
        "4 epochs；整个阶段不加载 Teacher。两条 policy 实验除初始化外配置完全"
        "一致。\n\n"
        "Policy objective 对每条 query 使用 Plackett–Luce 不放回采样产生 16 个"
        "独立 Top-10 "
        "rollouts。单个 rollout 内图片不重复；不同 rollouts 允许按照真实 policy 概率"
        "自然重复。Reward 是黄金 target 的单正样本 NDCG@10；advantage 使用同一 "
        "query 的组内 reward 均值和标准差进行标准化。\n\n"
        "## CIRR progression\n\n"
        + _table(
            ["Model", "Stage", "R@1", "R@5", "R@10", "R@50", "Group R@1"],
            curve_rows,
        )
        + "\n\n## Group-relative policy sampling diagnostics\n\n"
        + _table(
            [
                "Model",
                "Epoch",
                "Reward mean",
                "Reward std",
                "Effective query rate",
                "Unique rollouts",
                "Unique rollout rate",
                "Sampled target R@10",
                "Target rank incl. miss",
                "Target probability",
                "Policy entropy",
            ],
            policy_rows,
        )
        + "\n\n## Final checkpoint: all datasets\n\n"
        + _table(final_headers, final_table_rows)
        + "\n\n`—` 表示对应训练或评估尚未完成；重新运行总脚本会从已有 checkpoint "
        "续跑并补齐结果。\n"
    )


def _last_training_record_by_epoch(path: Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    if not path.is_file():
        return result
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            result[int(payload["epoch_number"])] = payload
    return result


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
