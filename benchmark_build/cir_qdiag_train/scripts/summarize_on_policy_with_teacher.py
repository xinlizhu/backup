#!/usr/bin/env python3
"""Combine the on-policy CLIP-L ablation and Teacher evaluation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLIP_ROOT = PROJECT_ROOT / "outputs/on_policy_live_top10/clip_l_original"
DEFAULT_TEACHER_ROOT = (
    PROJECT_ROOT / "outputs/on_policy_live_top10/teacher/BGE-VL-MLLM-S2"
)
DEFAULT_TEACHER_TRAIN = (
    PROJECT_ROOT
    / "outputs/on_policy_centered/cache/bge_mllm_s2_cirr_train/train_recall.json"
)

MODEL_SPECS = (
    ("CLIP-L-original", "baseline/val_metrics.json"),
    ("CLIP-L-OnPolicy-rank", "positive_rank/val_epoch_008.json"),
    ("CLIP-L-OnPolicy-centered", "centered_response/val_epoch_008.json"),
    ("CLIP-L-OnPolicy-joint", "joint/val_epoch_008.json"),
)
DIAGNOSTIC_SUBSETS = (
    "all",
    "strict",
    "add",
    "substitute",
    "delete",
    "object",
    "strict|object",
    "attribute",
    "strict|attribute",
    "predicate",
    "strict|predicate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-root", type=Path, default=DEFAULT_CLIP_ROOT)
    parser.add_argument("--teacher-root", type=Path, default=DEFAULT_TEACHER_ROOT)
    parser.add_argument(
        "--teacher-train-recall", type=Path, default=DEFAULT_TEACHER_TRAIN
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _require(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "required evaluation outputs are missing:\n" + "\n".join(missing)
        )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _markdown_table(frame: pd.DataFrame, formats: dict[str, str]) -> str:
    columns = frame.columns.tolist()
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---:" if c != "Model" and c != "Subset" else "---" for c in columns) + "|",
    ]
    for _, row in frame.iterrows():
        values: list[str] = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                rendered = "—"
            elif column in formats:
                rendered = formats[column].format(float(value))
            else:
                rendered = str(value)
            values.append(rendered.replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    clip_root = args.clip_root.resolve()
    teacher_root = args.teacher_root.resolve()
    output_dir = (args.output_dir or clip_root.parent).resolve()

    clip_official_path = clip_root / "pinpoint_official/official_map10_negrecall10.csv"
    teacher_official_path = (
        teacher_root / "pinpoint_official/official_map10_negrecall10.csv"
    )
    clip_diagnostic_path = clip_root / "pinpoint_diagnostics/combined_summary.csv"
    teacher_diagnostic_path = teacher_root / "pinpoint_diagnostics/combined_summary.csv"
    teacher_cirr_path = teacher_root / "cirr_val/val_metrics.json"
    cirr_paths = [clip_root / relative for _, relative in MODEL_SPECS]
    clip_official_detail_paths = [
        clip_root / "pinpoint_official" / model / "official_metrics.csv"
        for model, _ in MODEL_SPECS
    ]
    teacher_official_detail_path = (
        teacher_root
        / "pinpoint_official/BGE-VL-MLLM-S2/official_metrics.csv"
    )
    _require(
        cirr_paths
        + clip_official_detail_paths
        + [
            teacher_cirr_path,
            clip_official_path,
            teacher_official_path,
            teacher_official_detail_path,
            clip_diagnostic_path,
            teacher_diagnostic_path,
        ]
    )

    cirr_by_model: dict[str, dict] = {
        model: _read_json(path)
        for (model, _), path in zip(MODEL_SPECS, cirr_paths, strict=True)
    }
    cirr_by_model["BGE-VL-MLLM-S2 Teacher"] = _read_json(teacher_cirr_path)

    official = pd.concat(
        [pd.read_csv(clip_official_path), pd.read_csv(teacher_official_path)],
        ignore_index=True,
    )
    official["model"] = official["model"].replace(
        {"BGE-VL-MLLM-S2": "BGE-VL-MLLM-S2 Teacher"}
    )
    official_by_model = official.drop_duplicates("model", keep="last").set_index("model")

    model_order = [model for model, _ in MODEL_SPECS] + [
        "BGE-VL-MLLM-S2 Teacher"
    ]
    official_full = pd.concat(
        [
            *(pd.read_csv(path) for path in clip_official_detail_paths),
            pd.read_csv(teacher_official_detail_path),
        ],
        ignore_index=True,
    )
    official_full["model"] = official_full["model"].replace(
        {"BGE-VL-MLLM-S2": "BGE-VL-MLLM-S2 Teacher"}
    )
    official_full = (
        official_full.drop_duplicates("model", keep="last")
        .set_index("model")
        .loc[model_order]
        .reset_index()
        .rename(columns={"model": "Model"})
    )

    pinpoint_standard = official_full[
        [
            "Model",
            "mAP@1",
            "mAP@5",
            "mAP@10",
            "mAP@50",
            "precision@1",
            "precision@5",
            "precision@10",
            "precision@50",
            "recall@1",
            "recall@5",
            "recall@10",
            "recall@50",
        ]
    ]
    pinpoint_negative = official_full[
        [
            "Model",
            "NegRecall@1",
            "NegRecall@5",
            "NegRecall@10",
            "NegRecall@50",
            "mAP@1_noNeg",
            "mAP@5_noNeg",
            "mAP@10_noNeg",
            "mAP@50_noNeg",
            "delta_mAP@10_noNeg",
            "pct_increase_mAP@10_noNeg",
        ]
    ]
    pinpoint_linguistic = official_full[
        ["Model", "ling_sens_range", "ling_sens_std", "total_queries"]
    ]
    retrieval_rows: list[dict] = []
    for model in model_order:
        cirr = cirr_by_model[model]
        if model not in official_by_model.index:
            raise ValueError(f"PinPoint official metrics are absent for {model}")
        pin = official_by_model.loc[model]
        retrieval_rows.append(
            {
                "Model": model,
                "CIRR Queries": cirr["num_queries"],
                "CIRR R@1": cirr["recall_at_1"],
                "CIRR R@5": cirr["recall_at_5"],
                "CIRR R@10": cirr["recall_at_10"],
                "CIRR R@50": cirr["recall_at_50"],
                "Group R@1": cirr["group_recall_at_1"],
                "Group R@2": cirr["group_recall_at_2"],
                "Group R@3": cirr["group_recall_at_3"],
                "PinPoint mAP@10": pin["mAP@10"],
                "PinPoint NegRecall@10": pin["NegRecall@10"],
            }
        )
    retrieval = pd.DataFrame(retrieval_rows)

    diagnostics = pd.concat(
        [pd.read_csv(clip_diagnostic_path), pd.read_csv(teacher_diagnostic_path)],
        ignore_index=True,
    )
    diagnostics["model"] = diagnostics["model"].replace(
        {"BGE-VL-MLLM-S2": "BGE-VL-MLLM-S2 Teacher"}
    )
    selected = diagnostics.loc[
        diagnostics["model"].isin(model_order)
        & diagnostics["subset"].isin(DIAGNOSTIC_SUBSETS)
    ].copy()
    selected["model_order"] = selected["model"].map(
        {name: index for index, name in enumerate(model_order)}
    )
    selected["subset_order"] = selected["subset"].map(
        {name: index for index, name in enumerate(DIAGNOSTIC_SUBSETS)}
    )
    selected = selected.sort_values(["model_order", "subset_order"])
    expected_rows = len(model_order) * len(DIAGNOSTIC_SUBSETS)
    if len(selected) != expected_rows:
        pairs = set(zip(selected["model"], selected["subset"]))
        missing = [
            (model, subset)
            for model in model_order
            for subset in DIAGNOSTIC_SUBSETS
            if (model, subset) not in pairs
        ]
        raise ValueError(f"diagnostic summary is incomplete: {missing}")
    winrate = selected[
        [
            "model",
            "subset",
            "negative_count",
            "mean_target_delta",
            "target_raw_win_rate",
            "target_stable_win_rate",
            "normal_perturbation_threshold",
            "positive_query_own_minus_random_target",
        ]
    ].rename(
        columns={
            "model": "Model",
            "subset": "Subset",
            "negative_count": "Negatives",
            "mean_target_delta": "Mean target delta",
            "target_raw_win_rate": "Raw win rate",
            "target_stable_win_rate": "Stable win rate",
            "normal_perturbation_threshold": "tau95",
            "positive_query_own_minus_random_target": "Own target - random target",
        }
    )

    coverage = selected[
        [
            "model",
            "subset",
            "group_count",
            "instruction_count",
            "negative_count",
            "normal_perturbation_threshold",
            "threshold_bootstrap_ci_low",
            "threshold_bootstrap_ci_high",
            "target_raw_win_rate",
            "raw_win_rate_ci_low",
            "raw_win_rate_ci_high",
            "target_stable_win_rate",
            "stable_win_rate_ci_low",
            "stable_win_rate_ci_high",
        ]
    ].copy()
    coverage["tau95 95% CI"] = coverage.apply(
        lambda row: (
            f"[{row['threshold_bootstrap_ci_low']:.6f}, "
            f"{row['threshold_bootstrap_ci_high']:.6f}]"
        ),
        axis=1,
    )
    coverage["Raw Win 95% CI"] = coverage.apply(
        lambda row: (
            f"[{row['raw_win_rate_ci_low']:.6f}, "
            f"{row['raw_win_rate_ci_high']:.6f}]"
        ),
        axis=1,
    )
    coverage["Stable Win 95% CI"] = coverage.apply(
        lambda row: (
            f"[{row['stable_win_rate_ci_low']:.6f}, "
            f"{row['stable_win_rate_ci_high']:.6f}]"
        ),
        axis=1,
    )
    coverage = coverage[
        [
            "model",
            "subset",
            "group_count",
            "instruction_count",
            "negative_count",
            "normal_perturbation_threshold",
            "tau95 95% CI",
            "target_raw_win_rate",
            "Raw Win 95% CI",
            "target_stable_win_rate",
            "Stable Win 95% CI",
        ]
    ].rename(
        columns={
            "model": "Model",
            "subset": "Subset",
            "group_count": "Groups",
            "instruction_count": "Instructions",
            "negative_count": "Negatives",
            "normal_perturbation_threshold": "tau95",
            "target_raw_win_rate": "Raw win rate",
            "target_stable_win_rate": "Stable win rate",
        }
    )

    target_scores = selected[
        [
            "model",
            "subset",
            "mean_positive_query_own_target_score",
            "mean_negative_query_own_target_score",
            "mean_target_delta",
            "mean_positive_query_random_target_score",
            "positive_query_own_minus_random_target",
            "mean_negative_query_random_target_score",
            "negative_query_own_minus_random_target",
        ]
    ].rename(
        columns={
            "model": "Model",
            "subset": "Subset",
            "mean_positive_query_own_target_score": "Positive query -> own target",
            "mean_negative_query_own_target_score": "Negative query -> own target",
            "mean_target_delta": "Mean target delta",
            "mean_positive_query_random_target_score": "Positive query -> random target",
            "positive_query_own_minus_random_target": "Positive own - random target",
            "mean_negative_query_random_target_score": "Negative query -> random target",
            "negative_query_own_minus_random_target": "Negative own - random target",
        }
    )

    random_query_scores = selected[
        [
            "model",
            "subset",
            "mean_random_positive_query_own_target_score",
            "positive_query_own_minus_random_positive_query",
            "mean_random_negative_query_own_target_score",
            "negative_query_own_minus_random_negative_query",
        ]
    ].rename(
        columns={
            "model": "Model",
            "subset": "Subset",
            "mean_random_positive_query_own_target_score": "Random positive query -> own target",
            "positive_query_own_minus_random_positive_query": "Positive own - random positive query",
            "mean_random_negative_query_own_target_score": "Random negative query -> own target",
            "negative_query_own_minus_random_negative_query": "Negative own - random negative query",
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    retrieval.to_csv(output_dir / "ALL_MODELS_RETRIEVAL.csv", index=False)
    official_full.to_csv(
        output_dir / "ALL_MODELS_PINPOINT_OFFICIAL_FULL.csv", index=False
    )
    winrate.to_csv(output_dir / "ALL_MODELS_WINRATE.csv", index=False)
    selected.drop(columns=["model_order", "subset_order"]).to_csv(
        output_dir / "ALL_MODELS_DIAGNOSTICS_FULL.csv", index=False
    )

    retrieval_index = retrieval.set_index("Model")
    diag_index = selected.set_index(["model", "subset"])
    original = retrieval_index.loc["CLIP-L-original"]
    teacher = retrieval_index.loc["BGE-VL-MLLM-S2 Teacher"]
    rank = retrieval_index.loc["CLIP-L-OnPolicy-rank"]
    centered = retrieval_index.loc["CLIP-L-OnPolicy-centered"]
    joint = retrieval_index.loc["CLIP-L-OnPolicy-joint"]

    gains = pd.DataFrame(
        [
            {
                "Model": model,
                "Delta CIRR R@1": row["CIRR R@1"] - original["CIRR R@1"],
                "Delta CIRR R@10": row["CIRR R@10"] - original["CIRR R@10"],
                "Delta Group R@1": row["Group R@1"] - original["Group R@1"],
                "Delta PinPoint mAP@10": (
                    row["PinPoint mAP@10"] - original["PinPoint mAP@10"]
                ),
                "Relative mAP@10 gain": (
                    row["PinPoint mAP@10"] / original["PinPoint mAP@10"] - 1.0
                ),
                "NegRecall@10 reduction": (
                    original["PinPoint NegRecall@10"]
                    - row["PinPoint NegRecall@10"]
                ),
            }
            for model, row in retrieval_index.drop(index="CLIP-L-original").iterrows()
        ]
    )

    def diagnostic(model: str, subset: str, column: str) -> float:
        return float(diag_index.loc[(model, subset), column])

    parts = [
        "# On-Policy CLIP-L 与 BGE Teacher 完整评估汇总",
        "",
        (
            "本文汇总 CLIP-L 原版、三种 On-Policy KD 模型以及 BGE-VL-MLLM-S2 "
            "Teacher 的 CIRR val、PinPoint 官方检索和 PinPoint 细粒度诊断结果。"
        ),
        "",
        "## 指标口径",
        "",
        "- CIRR R@K、Group R@K 均为百分数（0–100，越高越好）；检索时排除 reference image。",
        "- PinPoint 官方指标使用全部 7,635 条 query，数值范围为 0–1。mAP、Precision、Recall 越高越好。",
        (
            "- PinPoint `NegRecall@K` 是 Top-K 中命中标注 negative 图片的比例，"
            "**越低越好**。它不是“成功排除 negative 的比例”。"
        ),
        (
            "- `strict` 排除 DELETE query；`strict|object/attribute/predicate` "
            "是在 strict 集合上按语义类型细分。"
        ),
        (
            "- 所有诊断均按 group 分层平衡；置信区间来自 2,000 次 group bootstrap。"
        ),
        "",
        "## 核心检索指标",
        "",
        _markdown_table(
            retrieval,
            {
                "CIRR Queries": "{:.0f}",
                "CIRR R@1": "{:.4f}",
                "CIRR R@5": "{:.4f}",
                "CIRR R@10": "{:.4f}",
                "CIRR R@50": "{:.4f}",
                "Group R@1": "{:.4f}",
                "Group R@2": "{:.4f}",
                "Group R@3": "{:.4f}",
                "PinPoint mAP@10": "{:.6f}",
                "PinPoint NegRecall@10": "{:.6f}",
            },
        ),
        "",
        "### 相对原始 CLIP-L 的变化",
        "",
        (
            "最后一列使用 `原版 NegRecall - 当前模型 NegRecall`，所以正值表示减少了"
            "错误 negative-image 命中。"
        ),
        "",
        _markdown_table(
            gains,
            {
                "Delta CIRR R@1": "{:+.4f}",
                "Delta CIRR R@10": "{:+.4f}",
                "Delta Group R@1": "{:+.4f}",
                "Delta PinPoint mAP@10": "{:+.6f}",
                "Relative mAP@10 gain": "{:+.2%}",
                "NegRecall@10 reduction": "{:+.6f}",
            },
        ),
        "",
        "## PinPoint 官方完整指标：mAP / Precision / Recall",
        "",
        _markdown_table(
            pinpoint_standard,
            {
                column: "{:.4f}"
                for column in pinpoint_standard.columns
                if column != "Model"
            },
        ),
        "",
        "## PinPoint 官方 negative 与 no-negative 指标",
        "",
        (
            "NegRecall 越低越好。`mAP@K_noNeg` 是先从排序结果中移除已知 "
            "negative 图片再计算 mAP；两者之差反映 negative 图片对正样本排序的干扰。"
        ),
        "",
        _markdown_table(
            pinpoint_negative,
            {
                column: "{:.4f}"
                for column in pinpoint_negative.columns
                if column != "Model"
            },
        ),
        "",
        "## PinPoint 官方语言敏感度",
        "",
        (
            "`ling_sens_range/std` 统计同一参考图不同文本表达之间 Precision@10 "
            "的波动；越低表示对改写越稳定，但应与检索精度一起判断。"
        ),
        "",
        _markdown_table(
            pinpoint_linguistic,
            {
                "ling_sens_range": "{:.4f}",
                "ling_sens_std": "{:.4f}",
                "total_queries": "{:.0f}",
            },
        ),
        "",
        "## 主要结论",
        "",
        (
            f"- 三个 Student 都显著提升 CIRR。Student 中 Joint 的 R@1 最高 "
            f"({joint['CIRR R@1']:.4f})，比原版高 "
            f"{joint['CIRR R@1'] - original['CIRR R@1']:.4f} 个百分点；Rank 的 "
            f"R@5/R@10/R@50 最好 ({rank['CIRR R@5']:.4f}/"
            f"{rank['CIRR R@10']:.4f}/{rank['CIRR R@50']:.4f})."
        ),
        (
            f"- Teacher 的 CIRR R@1/R@5/R@10 和全部 Group Recall 最强 "
            f"({teacher['CIRR R@1']:.4f}/{teacher['CIRR R@5']:.4f}/"
            f"{teacher['CIRR R@10']:.4f})；但 Rank 的 R@50 略高于 Teacher "
            f"({rank['CIRR R@50']:.4f} vs {teacher['CIRR R@50']:.4f})."
        ),
        (
            f"- Teacher 同时取得最高 PinPoint mAP@10 ({teacher['PinPoint mAP@10']:.6f}) "
            f"和最低、即最好的 NegRecall@10 ({teacher['PinPoint NegRecall@10']:.6f})。"
            f"Student 中 Centered 的 mAP@10 最高 ({centered['PinPoint mAP@10']:.6f})，"
            f"Rank 的 NegRecall@10 最低 ({rank['PinPoint NegRecall@10']:.6f})。"
        ),
        (
            f"- 在 strict negatives 上，Joint 的 Student Mean Delta 和 Stable Win "
            f"最高 ({diagnostic('CLIP-L-OnPolicy-joint', 'strict', 'mean_target_delta'):.6f}, "
            f"{diagnostic('CLIP-L-OnPolicy-joint', 'strict', 'target_stable_win_rate'):.6f})；"
            f"Centered 的 Student Raw Win 最高 "
            f"({diagnostic('CLIP-L-OnPolicy-centered', 'strict', 'target_raw_win_rate'):.6f})."
        ),
        (
            f"- 三个 Student 最明显的诊断短板是 DELETE：Stable Win 只有 "
            f"{diagnostic('CLIP-L-OnPolicy-centered', 'delete', 'target_stable_win_rate'):.6f}–"
            f"{diagnostic('CLIP-L-OnPolicy-rank', 'delete', 'target_stable_win_rate'):.6f}。"
            f"但这里评价的是 negative query 对目标图分数的影响，与官方 NegRecall 的 "
            f"negative-image 命中率不是同一个指标，不能直接互相解释。"
        ),
        (
            f"- 细粒度优势不同：Joint 的 strict Object Stable Win 最好 "
            f"({diagnostic('CLIP-L-OnPolicy-joint', 'strict|object', 'target_stable_win_rate'):.6f})；"
            f"Teacher 的 strict Attribute 最好 ({diagnostic('BGE-VL-MLLM-S2 Teacher', 'strict|attribute', 'target_stable_win_rate'):.6f})；"
            f"Rank 的 strict Predicate Stable Win 最好 ({diagnostic('CLIP-L-OnPolicy-rank', 'strict|predicate', 'target_stable_win_rate'):.6f})，"
            f"Centered 则有最高的 strict Predicate Raw Win "
            f"({diagnostic('CLIP-L-OnPolicy-centered', 'strict|predicate', 'target_raw_win_rate'):.6f})."
        ),
        (
            "- 各 Student/Teacher 的 strict Stable Win 95% CI 大量重叠，因此 Rank、"
            "Centered、Joint 在总体 strict 集合上的细小点估计差异不宜解释为确定性胜负。"
            "strict Object 上 Joint 与 Teacher 的 Stable Win CI 不重叠，是当前最清晰的"
            "细分类优势之一，但仍建议最终论文使用 paired significance test 验证。"
        ),
        (
            f"- 同义改写波动阈值 tau95：原版 {diagnostic('CLIP-L-original', 'all', 'normal_perturbation_threshold'):.6f}，"
            f"Rank {diagnostic('CLIP-L-OnPolicy-rank', 'all', 'normal_perturbation_threshold'):.6f}，"
            f"Centered {diagnostic('CLIP-L-OnPolicy-centered', 'all', 'normal_perturbation_threshold'):.6f}，"
            f"Joint {diagnostic('CLIP-L-OnPolicy-joint', 'all', 'normal_perturbation_threshold'):.6f}，"
            f"Teacher {diagnostic('BGE-VL-MLLM-S2 Teacher', 'all', 'normal_perturbation_threshold'):.6f}。"
            "阈值是每个 embedding 空间内部校准的，不能仅根据跨模型阈值大小判断谁更稳定。"
        ),
        "",
        "## PinPoint 细粒度诊断：全部子集",
        "",
        (
            "`Delta = sim(q+, target) - sim(q-, target)`；Raw Win 统计 Delta > 0；"
            "Stable Win 统计 Delta > tau95。tau95 由每个模型的同义改写分数波动单独"
            "校准。最后一列是正确目标与每条样本随机抽取 10 张无关目标之间的平均差距。"
        ),
        "",
        _markdown_table(
            winrate,
            {
                "Negatives": "{:.0f}",
                "Mean target delta": "{:.6f}",
                "Raw win rate": "{:.6f}",
                "Stable win rate": "{:.6f}",
                "tau95": "{:.6f}",
                "Own target - random target": "{:.6f}",
            },
        ),
        "",
        "## 样本覆盖与 Bootstrap 置信区间",
        "",
        (
            "每个子集均列出 group、instruction 和 negative 数量。区间是分层 "
            "group-bootstrap 95% CI；tau95 是模型级阈值，因此会在该模型各子集中重复。"
        ),
        "",
        _markdown_table(
            coverage,
            {
                "Groups": "{:.0f}",
                "Instructions": "{:.0f}",
                "Negatives": "{:.0f}",
                "tau95": "{:.6f}",
                "Raw win rate": "{:.6f}",
                "Stable win rate": "{:.6f}",
            },
        ),
        "",
        "## 相似度分数分解",
        "",
        (
            "以下均为分层 group-balanced 均值。每条 negative 固定随机抽取 10 张"
            "其他 group 的无关图片。不同模型的原始 cosine 标尺不同，因此应优先比较"
            "同一模型内的 Delta 或 gap，而不要直接比较跨模型绝对相似度。"
        ),
        "",
        _markdown_table(
            target_scores,
            {
                column: "{:.6f}"
                for column in target_scores.columns
                if column not in {"Model", "Subset"}
            },
        ),
        "",
        "## Random-query 对照",
        "",
        (
            "该对照为每条样本抽取其他 PinPoint group 的 10 条无关 positive/negative "
            "query，并与当前 group 的正确目标计算相似度。"
        ),
        "",
        _markdown_table(
            random_query_scores,
            {
                column: "{:.6f}"
                for column in random_query_scores.columns
                if column not in {"Model", "Subset"}
            },
        ),
    ]
    teacher_train_path = args.teacher_train_recall.resolve()
    if teacher_train_path.is_file():
        teacher_train = _read_json(teacher_train_path)
        parts.extend(
            [
                "",
                "## Teacher CIRR train 检索（仅供参考）",
                "",
                (
                    "这是此前已有的 CIRR train 全库结果；由于 split 和图库规模不同，"
                    "不能与上面的 val 表直接比较，只作为 Teacher sanity check。"
                ),
                "",
                (
                    f"R@1 {teacher_train['recall_at_1']:.4f}, "
                    f"R@5 {teacher_train['recall_at_5']:.4f}, "
                    f"R@10 {teacher_train['recall_at_10']:.4f}, "
                    f"R@32 {teacher_train['recall_at_32']:.4f}, "
                    f"R@50 {teacher_train['recall_at_50']:.4f}; "
                    f"mean rank {teacher_train['mean_rank']:.4f}, "
                    f"median rank {teacher_train['median_rank']:.4f}."
                ),
            ]
        )
    report_path = output_dir / "ALL_MODELS_RESULTS.md"
    report_path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"[done] {report_path}")


if __name__ == "__main__":
    main()
