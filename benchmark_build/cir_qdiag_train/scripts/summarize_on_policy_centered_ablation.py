#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


RUNS = {
    "CLIP-L-original": ("Original CLIP-L", "baseline/val_metrics.json", None),
    "CLIP-L-OnPolicy-rank": (
        "Positive Rank KD",
        "positive_rank/val_epoch_008.json",
        "positive_rank/on_policy/epoch_008/metadata.json",
    ),
    "CLIP-L-OnPolicy-centered": (
        "Centered Response KD",
        "centered_response/val_epoch_008.json",
        "centered_response/on_policy/epoch_008/metadata.json",
    ),
    "CLIP-L-OnPolicy-joint": (
        "Rank + Centered KD",
        "joint/val_epoch_008.json",
        "joint/on_policy/epoch_008/metadata.json",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    official = {
        row["model"]: row
        for row in _read_csv(root / "pinpoint_official/official_map10_negrecall10.csv")
    }
    diagnostics = {
        (row["model"], row["subset"]): row
        for row in _read_csv(root / "pinpoint_diagnostics/combined_summary.csv")
    }
    rows: list[dict[str, object]] = []
    for model_name, (label, cirr_relative, mining_relative) in RUNS.items():
        cirr = _read_json(root / cirr_relative)
        mining = {} if mining_relative is None else _read_json(root / mining_relative)
        strict = diagnostics.get((model_name, "strict"), {})
        strict_object = diagnostics.get((model_name, "strict|object"), {})
        rows.append(
            {
                "model": model_name,
                "experiment": label,
                **{
                    key: cirr.get(key)
                    for key in (
                        "recall_at_1",
                        "recall_at_5",
                        "recall_at_10",
                        "recall_at_50",
                        "group_recall_at_1",
                        "group_recall_at_2",
                        "group_recall_at_3",
                    )
                },
                "mAP@10": official.get(model_name, {}).get("mAP@10"),
                "NegRecall@10": official.get(model_name, {}).get("NegRecall@10"),
                "strict_mean_delta": strict.get("mean_target_delta"),
                "strict_raw_win": strict.get("target_raw_win_rate"),
                "strict_stable_win": strict.get("target_stable_win_rate"),
                "object_mean_delta": strict_object.get("mean_target_delta"),
                "object_raw_win": strict_object.get("target_raw_win_rate"),
                "object_stable_win": strict_object.get("target_stable_win_rate"),
                "stable_threshold": strict.get("normal_perturbation_threshold"),
                "own_minus_random": strict.get(
                    "positive_query_own_minus_random_target"
                ),
                "epoch8_raw_target_recall_at_32": mining.get(
                    "raw_target_recall_at_k"
                ),
                "epoch8_teacher_target_top1_rate": mining.get(
                    "teacher_target_top1_rate"
                ),
                "epoch8_teacher_target_top10_rate": mining.get(
                    "teacher_target_top10_rate"
                ),
            }
        )
    csv_path = root / "ablation_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# On-Policy Centered Distillation Ablation",
        "",
        "All trained variants start directly from ordinary pretrained CLIP-L. "
        "Each epoch mines the current Student Top-32; the ground-truth target is "
        "forced into candidate index 0. The epoch snapshot is used only for candidate "
        "selection: every selected image is re-encoded by the current Student in its "
        "minibatch. KD retains samples whose Teacher target rank within the selected "
        "32 candidates is at most 10. No QDiag negative query or InfoNCE is used.",
        "",
        "| Experiment | CIRR R@1 | R@5 | R@10 | Group R@1 | mAP@10 | NegRecall@10 | Object Delta | Object Raw Win | Object Stable Win | τ95 | Own−Random | Epoch8 raw Target R@32 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {experiment} | {r1} | {r5} | {r10} | {gr1} | {map10} | {neg10} | "
            "{delta} | {raw} | {stable} | {threshold} | {gap} | {target32} |".format(
                experiment=row["experiment"],
                r1=_number(row["recall_at_1"]),
                r5=_number(row["recall_at_5"]),
                r10=_number(row["recall_at_10"]),
                gr1=_number(row["group_recall_at_1"]),
                map10=_number(row["mAP@10"], 6),
                neg10=_number(row["NegRecall@10"], 6),
                delta=_number(row["object_mean_delta"], 6),
                raw=_number(row["object_raw_win"], 6),
                stable=_number(row["object_stable_win"], 6),
                threshold=_number(row["stable_threshold"], 6),
                gap=_number(row["own_minus_random"], 6),
                target32=_number(row["epoch8_raw_target_recall_at_32"], 6),
            )
        )
    (root / "ABLATION_SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _number(value: object, digits: int = 4) -> str:
    if value in (None, ""):
        return "—"
    return f"{float(value):.{digits}f}"


if __name__ == "__main__":
    main()
