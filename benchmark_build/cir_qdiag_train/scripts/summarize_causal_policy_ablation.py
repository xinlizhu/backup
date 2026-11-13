#!/usr/bin/env python3
"""Summarize causal-prefix REINFORCE, GRPO, and GSPO experiments."""

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
    run_dir: Path
    pinpoint_dir: Path
    qdiag_dir: Path
    fashioniq_dir: Path
    trained_here: bool


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/causal_policy_ablation_e4",
    )
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    specs = _specifications(root)
    rows = [_final_row(spec) for spec in specs]
    _write_csv(root / "all_metrics.csv", rows)
    destination = root / "RESULTS.md"
    destination.write_text(_render(specs, rows), encoding="utf-8")
    print(destination, flush=True)


def _specifications(root: Path) -> list[RunSpec]:
    opd = PROJECT_ROOT / "outputs/rank_candidate_policy_e4"
    old_rl = PROJECT_ROOT / "outputs/opd_listwise_rl_e4"
    return [
        RunSpec(
            "OPD E4 initialization",
            opd / "dynamic_student_rank",
            opd / "pinpoint_official/DynamicStudentRank",
            opd / "pinpoint_diagnostics/DynamicStudentRank",
            opd / "fashioniq/dynamic_student_rank",
            False,
        ),
        RunSpec(
            "Full-sequence REINFORCE (existing)",
            old_rl / "group_relative",
            old_rl / "pinpoint_official/CLIP-L-OPD-E4-GroupRelative",
            old_rl / "qdiag/CLIP-L-OPD-E4-GroupRelative",
            old_rl / "fashioniq/group_relative",
            False,
        ),
        RunSpec(
            "Causal-Prefix REINFORCE",
            root / "cp_reinforce",
            root / "pinpoint_official/CLIP-L-OPD-CP-Reinforce",
            root / "qdiag/CLIP-L-OPD-CP-Reinforce",
            root / "fashioniq/cp_reinforce",
            True,
        ),
        RunSpec(
            "Causal-Prefix GRPO",
            root / "cp_grpo",
            root / "pinpoint_official/CLIP-L-OPD-CP-GRPO",
            root / "qdiag/CLIP-L-OPD-CP-GRPO",
            root / "fashioniq/cp_grpo",
            True,
        ),
        RunSpec(
            "Causal-Prefix GSPO No-Clip",
            root / "cp_gspo_noclip",
            root / "pinpoint_official/CLIP-L-OPD-CP-GSPO-NoClip",
            root / "qdiag/CLIP-L-OPD-CP-GSPO-NoClip",
            root / "fashioniq/cp_gspo_noclip",
            True,
        ),
        RunSpec(
            "Causal-Prefix GSPO",
            root / "cp_gspo",
            root / "pinpoint_official/CLIP-L-OPD-CP-GSPO",
            root / "qdiag/CLIP-L-OPD-CP-GSPO",
            root / "fashioniq/cp_gspo",
            True,
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
    training_seconds = sum(
        float(row.get("total_seconds", 0.0))
        for row in timing.get("epochs_this_process", [])
    )
    return {
        "Model": spec.label,
        "Training hours": training_seconds / 3600.0 if training_seconds else None,
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
    }


def _render(specs: list[RunSpec], rows: list[dict[str, Any]]) -> str:
    cirr_curve: list[list[str]] = []
    policy_diagnostics: list[list[str]] = []
    for spec in specs:
        epochs = range(1, 5) if spec.trained_here else (4,)
        for epoch in epochs:
            metrics = _read_json(spec.run_dir / f"val_epoch_{epoch:03d}.json")
            if metrics:
                cirr_curve.append(
                    [
                        spec.label,
                        str(epoch),
                        _format(metrics.get("recall_at_1")),
                        _format(metrics.get("recall_at_5")),
                        _format(metrics.get("recall_at_10")),
                        _format(metrics.get("recall_at_50")),
                        _format(metrics.get("group_recall_at_1")),
                    ]
                )
        if spec.trained_here:
            for epoch, log in sorted(
                _last_training_record_by_epoch(
                    spec.run_dir / "train_metrics.jsonl"
                ).items()
            ):
                policy_diagnostics.append(
                    [
                        spec.label,
                        str(epoch),
                        _format(log.get("rollout_reward_mean")),
                        _format(log.get("effective_rl_query_rate")),
                        _format(log.get("causal_prefix_mean_length")),
                        _format(log.get("causal_suffix_masked_steps")),
                        _format(log.get("unique_reward_count")),
                        _format(log.get("sampled_target_recall_at_k")),
                        _format(log.get("policy_target_probability")),
                        _format(log.get("policy_entropy")),
                        _format(log.get("importance_ratio_mean")),
                        _format(log.get("clip_fraction")),
                        _format(log.get("approx_policy_kl")),
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
    ]
    return (
        "# Causal-prefix policy optimization ablation\n\n"
        "四个新实验从同一个 OPD CLIP-L E4 checkpoint 开始，使用相同的 "
        "epoch-level Student Top-32、16 rollouts、Target NDCG@10 reward、T=0.05、"
        "full finetune 和 seed 42。每条 rollout 在 target 首次出现时停止；若 target "
        "未进入前10，则有效长度为10。target 后不存在训练 suffix。\n\n"
        "CP-REINFORCE 每组更新一次；CP-GRPO、CP-GSPO No-Clip 和 CP-GSPO "
        "复用同一 rollout group 更新两次。GRPO 使用逐位置 ratio；GSPO 使用有效前缀"
        "的长度归一化 sequence ratio。\n\n"
        "## CIRR checkpoint curve\n\n"
        + _table(
            ["Model", "Epoch", "R@1", "R@5", "R@10", "R@50", "Group R@1"],
            cirr_curve,
        )
        + "\n\n## Causal policy diagnostics\n\n"
        + _table(
            [
                "Model",
                "Epoch",
                "Reward",
                "Effective groups",
                "Prefix length",
                "Masked suffix",
                "Unique rewards",
                "Sampled target R@10",
                "Target probability",
                "Entropy",
                "Ratio mean",
                "Clip fraction",
                "Approx KL",
            ],
            policy_diagnostics,
        )
        + "\n\n## Final checkpoint: all datasets\n\n"
        + _table(
            final_headers,
            [[_format(row.get(header)) for header in final_headers] for row in rows],
        )
        + "\n\n`—` 表示对应训练或评估尚未完成；重新运行总脚本会从最近完成的 "
        "epoch checkpoint 继续并补齐评估。\n"
    )


def _last_training_record_by_epoch(path: Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    if not path.is_file():
        return result
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
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
