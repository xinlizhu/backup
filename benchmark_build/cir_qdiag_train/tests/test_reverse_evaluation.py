from __future__ import annotations

import pytest
import torch

from cir_qdiag_train.reverse_evaluation import compute_reverse_metrics


def test_reverse_metrics_treat_shared_target_queries_as_multiple_positives() -> None:
    # Query rows 0 and 2 belong to target 0; rows 1 and 3 belong to target 1.
    scores = torch.tensor(
        [
            [0.9, 0.8, 0.7, 0.6],
            [0.4, 0.9, 0.3, 0.8],
        ]
    )
    metrics = compute_reverse_metrics(
        scores,
        torch.tensor([0, 1, 0, 1]),
        cutoffs=[1, 2],
    )

    assert metrics["num_reverse_targets"] == 2
    assert metrics["targets_with_multiple_queries"] == 2
    assert metrics["target_hit_rate_at_1"] == 100.0
    assert metrics["macro_query_recall_at_1"] == 50.0
    assert metrics["pair_recall_at_1"] == 50.0
    assert metrics["map_at_1"] == 1.0
    assert metrics["map_at_2"] == 0.75
    assert metrics["mrr"] == 1.0
    assert metrics["mean_positive_rank"] == 1.75


def test_reverse_metrics_reject_target_without_a_positive_query() -> None:
    with pytest.raises(ValueError, match="every reverse target"):
        compute_reverse_metrics(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([0, 0]),
            cutoffs=[1],
        )
