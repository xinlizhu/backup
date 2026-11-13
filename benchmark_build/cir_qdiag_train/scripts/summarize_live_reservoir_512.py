#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


RUNS = (
    ("grpo_no_anchor_kl", "R512 GRPO"),
    ("grpo_adaptive_anchor_kl", "R512 GRPO + adaptive anchor KL"),
    ("gspo_no_anchor_kl", "R512 GSPO"),
    ("gspo_adaptive_anchor_kl", "R512 GSPO + adaptive anchor KL"),
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
        if not line.strip():
            continue
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
            epoch_tag = f"{epoch:03d}"
            cirr = _read_json(run_dir / f"val_epoch_{epoch_tag}.json")
            fashion = _read_json(
                root
                / "fashioniq"
                / run_name
                / f"epoch_{epoch_tag}"
                / "metrics.json"
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
                    "Live Raw Target R@32": train.get(
                        "live_reservoir_raw_target_recall_at_k"
                    ),
                    "Live Target Mean Rank@512": train.get(
                        "live_reservoir_target_mean_rank"
                    ),
                    "Anchor KL": train.get("anchor_policy_kl"),
                    "Anchor KL EMA": train.get("anchor_kl_ema"),
                    "Anchor KL Beta": train.get(
                        "anchor_kl_beta_after_update"
                    ),
                    "Reward": train.get("rollout_reward_mean"),
                    "Effective Query Rate": train.get(
                        "effective_rl_query_rate"
                    ),
                    "Clip Fraction": train.get("clip_fraction"),
                }
            )

    fields = list(rows[0])
    with (root / "all_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
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
        "Anchor KL",
        "Anchor KL Beta",
    ]
    lines = [
        "# Live Top-512 reservoir policy results",
        "",
        "Teacher-free CLIP-L full finetuning from the same OPD initialization. "
        "The epoch-start Student builds an ephemeral Top-512 reservoir; the latest "
        "minibatch policy screens it to target plus 31 current hard images, followed "
        "by 16 target-NDCG causal rollouts and two GRPO/GSPO updates. Reservoirs, "
        "anchor scores, and screening features are memory-only and released after use.",
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
            "Full diagnostics, including live raw target R@32, target mean rank in "
            "the reservoir, reward, effective-query rate, clipping, and adaptive-KL "
            "state, are stored in `all_metrics.csv`.",
            "",
        ]
    )
    (root / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
