from __future__ import annotations

from collections import defaultdict
from statistics import mean, median
from typing import Any

from cir_qdiag.schemas import ConstructionMode, EditOperation, SemanticTarget


def evaluate_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    selection_values: list[bool] = []
    hardest_margins: list[float] = []
    for row in rows:
        positive = float(row["positive_score"])
        negatives = row.get("negative_scores", {})
        labels = row.get("edit_labels", {})
        margins: list[float] = []
        for negative_id, score_value in negatives.items():
            label = labels[negative_id]
            score = float(score_value)
            margin = positive - score
            margins.append(margin)
            pairs.append(
                {
                    "sample_id": row["sample_id"],
                    "negative_id": negative_id,
                    **label,
                    "positive_score": positive,
                    "negative_score": score,
                    "margin": margin,
                    "correct": positive > score,
                }
            )
        if negatives:
            selection_values.append(positive > max(float(value) for value in negatives.values()))
            hardest_margins.append(min(margins))

    overall = _pair_metrics(pairs)
    overall["mean_hardest_negative_margin"] = mean(hardest_margins) if hardest_margins else 0.0
    overall["median_hardest_negative_margin"] = (
        median(hardest_margins) if hardest_margins else 0.0
    )
    by_target = _group_metrics(pairs, "semantic_target", [item.value for item in SemanticTarget])
    by_operation = _group_metrics(
        pairs, "edit_operation", [item.value for item in EditOperation]
    )
    by_mode = _group_metrics(
        pairs, "construction_mode", [item.value for item in ConstructionMode]
    )
    by_cell = _cell_metrics(pairs)
    populated = [value["pairwise_accuracy"] for value in by_cell.values() if value["count"]]
    return {
        "overall": overall,
        "by_semantic_target": by_target,
        "by_edit_operation": by_operation,
        "by_construction_mode": by_mode,
        "by_edit_cell_and_mode": by_cell,
        "macro_edit_cell_accuracy": mean(populated) if populated else 0.0,
        "modification_selection_accuracy": (
            mean(selection_values) if selection_values else 0.0
        ),
        "selection_sample_count": len(selection_values),
        "target_leakage_strict_non_deletion": _leakage_metrics(rows),
    }


def _group_metrics(
    pairs: list[dict[str, Any]], key: str, values: list[str]
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in pairs:
        grouped[item[key]].append(item)
    return {value: _pair_metrics(grouped[value]) for value in values}


def _cell_metrics(pairs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in pairs:
        key = (
            f"{item['semantic_target']}|{item['edit_operation']}|"
            f"{item['construction_mode']}"
        )
        grouped[key].append(item)
    return {
        f"{target.value}|{operation.value}|{mode.value}": _pair_metrics(
            grouped[f"{target.value}|{operation.value}|{mode.value}"]
        )
        for target in SemanticTarget
        for operation in EditOperation
        for mode in ConstructionMode
    }


def _pair_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {
            "count": 0,
            "pairwise_accuracy": 0.0,
            "mean_margin": 0.0,
            "median_margin": 0.0,
            "negative_margin_rate": 0.0,
        }
    margins = [item["margin"] for item in items]
    return {
        "count": len(items),
        "pairwise_accuracy": mean(item["correct"] for item in items),
        "mean_margin": mean(margins),
        "median_margin": median(margins),
        "negative_margin_rate": mean(value <= 0 for value in margins),
    }


def _leakage_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranks: list[int] = []
    for row in rows:
        labels = row.get("edit_labels", {})
        for negative_id, rank in row.get("target_ranks", {}).items():
            label = labels.get(negative_id, {})
            if label.get("edit_operation") != EditOperation.DELETE.value and rank is not None:
                ranks.append(int(rank))
    if not ranks:
        return {
            "count": 0,
            "leakage_at_1": None,
            "leakage_at_5": None,
            "leakage_at_10": None,
            "median_rank": None,
        }
    return {
        "count": len(ranks),
        "leakage_at_1": mean(rank <= 1 for rank in ranks),
        "leakage_at_5": mean(rank <= 5 for rank in ranks),
        "leakage_at_10": mean(rank <= 10 for rank in ranks),
        "median_rank": median(ranks),
    }
