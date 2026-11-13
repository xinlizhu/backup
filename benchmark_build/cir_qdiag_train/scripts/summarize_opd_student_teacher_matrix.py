#!/usr/bin/env python3
"""Summarize the cross-Student/cross-Teacher Rank-OPD matrix."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ResultSpec:
    model: str
    role: str
    student: str
    teacher: str
    cirr: Path | None = None
    fashioniq: Path | None = None
    pinpoint: Path | None = None
    qdiag: Path | None = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/opd_student_teacher_matrix",
    )
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    specs = _specifications(output_root)
    rows = [_read_result(spec) for spec in specs]
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "all_metrics.csv", index=False)
    (output_root / "RESULTS.md").write_text(
        _render_markdown(frame), encoding="utf-8"
    )
    print(output_root / "RESULTS.md", flush=True)


def _specifications(root: Path) -> list[ResultSpec]:
    live = PROJECT_ROOT / "outputs/on_policy_live_top10"
    raw_qwen = PROJECT_ROOT / "outputs/qwen3_vl_original_baselines"
    fashion = PROJECT_ROOT / "outputs/fashioniq_val"
    return [
        ResultSpec(
            "BGE-VL-MLLM-S2",
            "Teacher reference",
            "—",
            "—",
            live / "teacher/BGE-VL-MLLM-S2/cirr_val/val_metrics.json",
            fashion / "bge_vl_mllm_s2_teacher/metrics.json",
            live
            / "teacher/BGE-VL-MLLM-S2/pinpoint_official/BGE-VL-MLLM-S2/official_metrics.csv",
            live
            / "teacher/BGE-VL-MLLM-S2/pinpoint_diagnostics/BGE-VL-MLLM-S2/summary.csv",
        ),
        ResultSpec(
            "Qwen3-VL-Embedding-8B",
            "Teacher reference",
            "—",
            "—",
            raw_qwen / "cirr/Qwen3-VL-Embedding-8B/val_metrics.json",
            raw_qwen / "fashioniq/Qwen3-VL-Embedding-8B/metrics.json",
            raw_qwen
            / "pinpoint_official/Qwen3-VL-Embedding-8B/official_metrics.csv",
            raw_qwen / "qdiag/Qwen3-VL-Embedding-8B/summary.csv",
        ),
        ResultSpec(
            "CLIP-B original",
            "Student baseline",
            "CLIP-B",
            "—",
            root / "clip_b_original/cirr_val.json",
            root / "fashioniq/clip_b_original/metrics.json",
            root / "pinpoint_official/CLIP-B-original/official_metrics.csv",
            root / "qdiag/CLIP-B-original/summary.csv",
        ),
        ResultSpec(
            "CLIP-L original",
            "Student baseline",
            "CLIP-L",
            "—",
            live / "clip_l_original/baseline/val_metrics.json",
            fashion / "clip_l_original/metrics.json",
            live
            / "clip_l_original/pinpoint_official/CLIP-L-original/official_metrics.csv",
            live
            / "clip_l_original/pinpoint_diagnostics/CLIP-L-original/summary.csv",
        ),
        ResultSpec(
            "Qwen3-VL-Embedding-2B original",
            "Student baseline",
            "Qwen-2B",
            "—",
            raw_qwen / "cirr/Qwen3-VL-Embedding-2B/val_metrics.json",
            raw_qwen / "fashioniq/Qwen3-VL-Embedding-2B/metrics.json",
            raw_qwen
            / "pinpoint_official/Qwen3-VL-Embedding-2B/official_metrics.csv",
            raw_qwen / "qdiag/Qwen3-VL-Embedding-2B/summary.csv",
        ),
        ResultSpec(
            "CLIP-L ← S2",
            "Existing OPD",
            "CLIP-L",
            "BGE-S2",
            PROJECT_ROOT
            / "outputs/rank_candidate_policy_e4/dynamic_student_rank/val_epoch_004.json",
            PROJECT_ROOT
            / "outputs/rank_candidate_policy_e4/fashioniq/dynamic_student_rank/metrics.json",
            PROJECT_ROOT
            / "outputs/rank_candidate_policy_e4/pinpoint_official/DynamicStudentRank/official_metrics.csv",
            PROJECT_ROOT
            / "outputs/rank_candidate_policy_e4/pinpoint_diagnostics/DynamicStudentRank/summary.csv",
        ),
        ResultSpec(
            "CLIP-B ← S2",
            "New OPD",
            "CLIP-B",
            "BGE-S2",
            root / "clip_b_from_s2/val_epoch_004.json",
            root / "fashioniq/clip_b_from_s2/metrics.json",
            root / "pinpoint_official/CLIP-B-OPD-S2/official_metrics.csv",
            root / "qdiag/CLIP-B-OPD-S2/summary.csv",
        ),
        ResultSpec(
            "CLIP-B ← Qwen-8B",
            "New OPD",
            "CLIP-B",
            "Qwen-8B",
            root / "clip_b_from_qwen8/val_epoch_004.json",
            root / "fashioniq/clip_b_from_qwen8/metrics.json",
            root / "pinpoint_official/CLIP-B-OPD-Qwen8/official_metrics.csv",
            root / "qdiag/CLIP-B-OPD-Qwen8/summary.csv",
        ),
        ResultSpec(
            "CLIP-L ← Qwen-8B",
            "New OPD",
            "CLIP-L",
            "Qwen-8B",
            root / "clip_l_from_qwen8/val_epoch_004.json",
            root / "fashioniq/clip_l_from_qwen8/metrics.json",
            root / "pinpoint_official/CLIP-L-OPD-Qwen8/official_metrics.csv",
            root / "qdiag/CLIP-L-OPD-Qwen8/summary.csv",
        ),
        ResultSpec(
            "Qwen-2B query-LoRA ← Qwen-8B",
            "New OPD",
            "Qwen-2B query-LoRA",
            "Qwen-8B",
            root / "qwen2_from_qwen8/cirr_val/val_metrics.json",
            root / "fashioniq/qwen2_from_qwen8/metrics.json",
            root
            / "pinpoint_official/Qwen2-QueryLoRA-OPD-Qwen8/official_metrics.csv",
            root / "qdiag/Qwen2-QueryLoRA-OPD-Qwen8/summary.csv",
        ),
    ]


def _read_result(spec: ResultSpec) -> dict[str, object]:
    row: dict[str, object] = {
        "Model": spec.model,
        "Role": spec.role,
        "Student": spec.student,
        "Teacher": spec.teacher,
    }
    if spec.cirr is not None and spec.cirr.is_file():
        values = json.loads(spec.cirr.read_text(encoding="utf-8"))
        for source, target in (
            ("recall_at_1", "CIRR R@1"),
            ("recall_at_5", "CIRR R@5"),
            ("recall_at_10", "CIRR R@10"),
            ("group_recall_at_1", "CIRR Group R@1"),
        ):
            row[target] = values.get(source)
    if spec.fashioniq is not None and spec.fashioniq.is_file():
        values = json.loads(spec.fashioniq.read_text(encoding="utf-8"))
        macro = values.get("macro_average", {})
        row["FashionIQ R@10"] = macro.get("recall_at_10")
        row["FashionIQ R@50"] = macro.get("recall_at_50")
    if spec.pinpoint is not None and spec.pinpoint.is_file():
        values = pd.read_csv(spec.pinpoint).iloc[0]
        row["PinPoint mAP@10"] = values.get("mAP@10")
        row["PinPoint NegRecall@10"] = values.get("NegRecall@10")
    if spec.qdiag is not None and spec.qdiag.is_file():
        values = pd.read_csv(spec.qdiag)
        strict = values.loc[values["subset"] == "strict"]
        if not strict.empty:
            values = strict.iloc[0]
            row["QDiag Delta"] = values.get("mean_target_delta")
            row["QDiag Raw Win"] = values.get("target_raw_win_rate")
            row["QDiag Stable Win"] = values.get("target_stable_win_rate")
            row["QDiag tau95"] = values.get("normal_perturbation_threshold")
            row["QDiag Own-Random"] = values.get(
                "positive_query_own_minus_random_target"
            )
    row["Complete"] = all(
        path is None or path.is_file()
        for path in (spec.cirr, spec.fashioniq, spec.pinpoint, spec.qdiag)
    )
    return row


def _render_markdown(frame: pd.DataFrame) -> str:
    columns = [
        "Model",
        "Student",
        "Teacher",
        "CIRR R@1",
        "CIRR R@10",
        "CIRR Group R@1",
        "FashionIQ R@10",
        "FashionIQ R@50",
        "PinPoint mAP@10",
        "PinPoint NegRecall@10",
        "QDiag Delta",
        "QDiag Raw Win",
        "QDiag Stable Win",
        "QDiag tau95",
        "QDiag Own-Random",
    ]
    selected = frame.reindex(columns=columns)
    table_rows = []
    for _, row in selected.iterrows():
        table_rows.append(
            [
                str(value)
                if isinstance(value, str)
                else ("—" if pd.isna(value) else f"{float(value):.4f}")
                for value in row
            ]
        )
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = "\n".join("| " + " | ".join(row) + " |" for row in table_rows)
    return (
        "# OPD Student–Teacher matrix\n\n"
        "本表只比较 Rank-OPD。所有新 CLIP 实验从原始 CLIP 权重开始，采用 "
        "Student 动态 Top-32、强制包含 target、Teacher target 在候选 Top-10 内的样本"
        "参与蒸馏，并训练 4 epoch、每轮保存。现有及本轮主矩阵使用 forward KL "
        "`D_KL(p_teacher || p_student)`。\n\n"
        "Qwen-2B Student 是非对称 query-LoRA：原始 image encoder/图库固定，query policy "
        "每轮重挖 Top-K；因此它应和自身原始基线比较，不把训练形式与 CLIP Full "
        "finetune 混为同一个变量。\n\n"
        + header
        + "\n"
        + divider
        + "\n"
        + body
        + "\n\n"
        "`—` 表示对应评估尚未产生；再次运行总脚本会从 cache/checkpoint 续跑并补齐。\n"
    )


if __name__ == "__main__":
    main()
