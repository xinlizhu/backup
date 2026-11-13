#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


RUNS = (
    ("reverse_opd", "CLIP-L Reward-Tilted Reverse-OPD"),
    ("reverse_gspo", "CLIP-L Reward-Tilted Reverse-KL GSPO"),
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _training_logs(path: Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    if not path.is_file():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            result[int(row["epoch_number"])] = row
    return result


def _format(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for run_name, model_name in RUNS:
        run_dir = root / run_name
        logs = _training_logs(run_dir / "train_metrics.jsonl")
        for epoch in range(1, 5):
            tag = f"{epoch:03d}"
            cirr = _read_json(run_dir / f"val_epoch_{tag}.json")
            fashion = _read_json(
                root / "fashioniq" / run_name / f"epoch_{tag}" / "metrics.json"
            ).get("macro_average", {})
            train = logs.get(epoch, {})
            rows.append(
                {
                    "Model": model_name,
                    "Epoch": epoch,
                    "CIRR R@1": cirr.get("recall_at_1"),
                    "CIRR R@5": cirr.get("recall_at_5"),
                    "CIRR R@10": cirr.get("recall_at_10"),
                    "CIRR R@50": cirr.get("recall_at_50"),
                    "Group R@1": cirr.get("group_recall_at_1"),
                    "Group R@2": cirr.get("group_recall_at_2"),
                    "Group R@3": cirr.get("group_recall_at_3"),
                    "FashionIQ R@10": fashion.get("recall_at_10"),
                    "FashionIQ R@50": fashion.get("recall_at_50"),
                    "Raw Target R@32": train.get("raw_mining_target_recall_at_k"),
                    "NDCG Reward": train.get("rollout_reward_mean"),
                    "Tilted Reward": train.get("objective_reward_mean"),
                    "Effective Query Rate": train.get("effective_rl_query_rate"),
                    "Teacher Reverse-KL Sample": train.get(
                        "teacher_reverse_kl_sample"
                    ),
                    "Tilted Target ESS": train.get(
                        "tilted_target_effective_sample_size"
                    ),
                    "Tilted Target Max Probability": train.get(
                        "tilted_target_max_probability"
                    ),
                    "Valid Teacher Rate": train.get("valid_teacher_rate"),
                    "Clip Fraction": train.get("clip_fraction"),
                }
            )

    fields = list(rows[0])
    with (root / "all_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    headline = [
        "Model",
        "Epoch",
        "CIRR R@1",
        "CIRR R@10",
        "Group R@1",
        "FashionIQ R@10",
        "FashionIQ R@50",
    ]
    lines = [
        "# NDCG-Tilted Reverse-KL experiments",
        "",
        "Both models full-finetune the original CLIP-L/14 for four epochs on the "
        "same dynamic Student Top-32 candidates, with the gold target forced into "
        "the set and 16 target-stopped Top-10 rollouts. BGE-VL-MLLM-S2 similarities "
        "are converted to probability distributions at T=0.05; target NDCG applies "
        "an exponential tilt with beta=0.1. Reverse-OPD directly minimizes group "
        "reverse KL and retains the Teacher target Top-10 filter. Reverse-KL GSPO "
        "uses the equivalent shaped reward with two sequence-level clipped updates, "
        "no Teacher filter, and no additional anchor KL.",
        "",
        "| " + " | ".join(headline) + " |",
        "| " + " | ".join(["---"] * len(headline)) + " |",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(_format(row[field]) for field in headline) + " |"
        )
    lines.extend(
        [
            "",
            "Training diagnostics, including reward, Teacher reverse-KL estimate, "
            "tilted-target ESS, valid-Teacher rate and GSPO clipping, are stored in "
            "`all_metrics.csv`.",
            "",
        ]
    )
    (root / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
