from __future__ import annotations

import importlib
from typing import Any

from cir_qdiag.schemas import SampleRecord


def load_scorer(spec: str):
    if ":" not in spec:
        raise ValueError("scorer must be specified as module:attribute")
    module_name, attribute = spec.split(":", 1)
    value = getattr(importlib.import_module(module_name), attribute)
    return value() if isinstance(value, type) else value


def score_records(records: list[SampleRecord], scorer: Any) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        positive_score = float(
            scorer.score(
                record.reference.image_path, record.positive_text, record.target.image_path
            )
        )
        negative_scores = {}
        target_ranks = {}
        edit_labels = {}
        for negative in record.negatives:
            negative_scores[negative.negative_id] = float(
                scorer.score(record.reference.image_path, negative.text, record.target.image_path)
            )
            edit_labels[negative.negative_id] = {
                "semantic_target": negative.semantic_target.value,
                "edit_operation": negative.edit_operation.value,
                "construction_mode": negative.construction_mode.value,
            }
            if hasattr(scorer, "target_rank"):
                target_ranks[negative.negative_id] = int(
                    scorer.target_rank(
                        record.reference.image_path, negative.text, record.target.image_path
                    )
                )
        rows.append(
            {
                "sample_id": record.sample_id,
                "positive_score": positive_score,
                "negative_scores": negative_scores,
                "edit_labels": edit_labels,
                "target_ranks": target_ranks,
            }
        )
    return rows
