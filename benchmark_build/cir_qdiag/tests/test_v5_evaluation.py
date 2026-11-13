from cir_qdiag.evaluation.metrics import evaluate_scores


def test_metrics_group_by_all_three_v5_axes() -> None:
    metrics = evaluate_scores(
        [
            {
                "sample_id": "x",
                "positive_score": 0.8,
                "negative_scores": {"n": 0.4},
                "edit_labels": {
                    "n": {
                        "semantic_target": "object",
                        "edit_operation": "add",
                        "construction_mode": "vision_grounded",
                    }
                },
                "target_ranks": {"n": 3},
            }
        ]
    )
    assert metrics["overall"]["pairwise_accuracy"] == 1.0
    assert metrics["by_semantic_target"]["object"]["count"] == 1
    assert metrics["by_edit_operation"]["add"]["count"] == 1
    assert metrics["by_construction_mode"]["vision_grounded"]["count"] == 1
    assert len(metrics["by_edit_cell_and_mode"]) == 18
