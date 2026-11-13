from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from counterfactual_profile_analysis import (  # noqa: E402
    MASK_COLUMNS,
    PERMUTED_MASK_COLUMNS,
    PERMUTED_RESPONSE_COLUMNS,
    RESPONSE_COLUMNS,
    ModelArtifacts,
    ModelSpec,
    binary_ndcg_at_k,
    build_shared_union_pool,
    candidate_shuffle_within_query,
    clean_identifier_values,
    cross_model_positive_query_rankings,
    default_feature_specs,
    make_group_fold_map,
    official_ap_at_k,
    official_key,
    official_negative_recall_at_k,
    paired_model_incremental_probe_did,
    paired_model_probe_statistics,
    positive_recall_at_k,
    query_balanced_class_weights,
    ranking_set_jaccard,
    rerank_native_prefix,
    summarize_candidate_responses,
    summarize_positive_paraphrases,
)


def synthetic_model_artifact(
    name: str,
    signatures: list[str],
    query_ids: list[str],
    top_indices: list[list[int]],
) -> ModelArtifacts:
    signature_array = np.asarray(signatures, dtype=object)
    query_keys = np.asarray([official_key(value) for value in query_ids], dtype=object)
    rankings = np.asarray(top_indices, dtype=np.int64)
    return ModelArtifacts(
        spec=ModelSpec(name, Path(f"/{name}.ckpt"), Path(f"/{name}-cache")),
        corpus=np.zeros((len(signatures), 2), dtype=np.float32),
        signatures=signature_array,
        signature_to_index={value: index for index, value in enumerate(signatures)},
        positive_queries=np.zeros((len(query_ids), 2), dtype=np.float32),
        query_keys=query_keys,
        key_to_index={str(value): index for index, value in enumerate(query_keys)},
        top_scores=np.zeros(rankings.shape, dtype=np.float32),
        top_indices=rankings,
    )


def test_candidate_responses_average_within_category_then_across_sparse_categories() -> (
    None
):
    summary = summarize_candidate_responses(
        positive_query=np.asarray([1.0], dtype=np.float32),
        candidate_embeddings=np.asarray([[1.0]], dtype=np.float32),
        negative_embeddings=np.asarray([[0.0], [0.2], [0.6], [0.8]], dtype=np.float32),
        negative_categories=[
            "object_add",
            "object_add",
            "attribute_substitute",
            "attribute_substitute",
        ],
    )

    # Edit deltas are [1.0, 0.8, 0.4, 0.2].  Categories first collapse to
    # 0.9 and 0.3, so category extrema differ from edit-level extrema.
    assert summary["sim_positive"] == pytest.approx([1.0])
    assert summary["d_edit_micro"] == pytest.approx([0.6])
    assert summary["d_object_add"] == pytest.approx([0.9])
    assert summary["d_attribute_substitute"] == pytest.approx([0.3])
    assert summary["d_available"] == pytest.approx([0.6])
    assert summary["d_object"] == pytest.approx([0.9])
    assert summary["d_attribute"] == pytest.approx([0.3])
    assert summary["d_add"] == pytest.approx([0.9])
    assert summary["d_substitute"] == pytest.approx([0.3])
    assert summary["d_min"] == pytest.approx([0.3])
    assert summary["d_max"] == pytest.approx([0.9])
    assert summary["d_edit_min"] == pytest.approx([0.2])
    assert summary["d_edit_max"] == pytest.approx([1.0])

    assert np.isnan(summary["d_predicate"]).all()
    for column in (
        "d_object_substitute",
        "d_attribute_add",
        "d_predicate_add",
        "d_predicate_substitute",
    ):
        assert np.isnan(summary[column]).all()


def test_candidate_responses_without_negatives_return_nan_diagnostics() -> None:
    candidates = np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    summary = summarize_candidate_responses(
        positive_query=np.asarray([0.5, 0.25], dtype=np.float32),
        candidate_embeddings=candidates,
        negative_embeddings=np.empty((0, 2), dtype=np.float32),
        negative_categories=[],
    )

    assert summary["sim_positive"] == pytest.approx([0.5, 0.5])
    diagnostic_columns = {
        "d_edit_micro",
        "d_edit_min",
        "d_edit_max",
        "d_edit_std",
        "d_available",
        "d_min",
        "d_max",
        "d_std",
        "d_object",
        "d_attribute",
        "d_predicate",
        "d_add",
        "d_substitute",
        *RESPONSE_COLUMNS,
    }
    assert diagnostic_columns == set(summary) - {"sim_positive"}
    assert all(np.isnan(summary[column]).all() for column in diagnostic_columns)


def test_positive_paraphrases_exclude_source_from_delta_summary() -> None:
    summary = summarize_positive_paraphrases(
        positive_query=np.asarray([1.0, 0.0], dtype=np.float32),
        group_positive_queries=np.asarray(
            [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
            dtype=np.float32,
        ),
        candidate_embeddings=np.asarray(
            [[1.0, 0.0], [0.0, 1.0]],
            dtype=np.float32,
        ),
        source_position=1,
    )

    assert summary["sim_positive_ensemble"] == pytest.approx([2.0 / 3.0] * 2)
    assert summary["paraphrase_delta_mean"] == pytest.approx([0.5, -1.0])
    assert summary["paraphrase_delta_std"] == pytest.approx([0.5, 0.0])


def test_positive_paraphrases_singleton_has_nan_delta_summary() -> None:
    summary = summarize_positive_paraphrases(
        positive_query=np.asarray([1.0, 0.0], dtype=np.float32),
        group_positive_queries=np.asarray([[1.0, 0.0]], dtype=np.float32),
        candidate_embeddings=np.asarray([[0.25, 1.0]], dtype=np.float32),
        source_position=0,
    )

    assert summary["sim_positive_ensemble"] == pytest.approx([0.25])
    assert np.isnan(summary["paraphrase_delta_mean"]).all()
    assert np.isnan(summary["paraphrase_delta_std"]).all()


def test_candidate_shuffle_is_deterministic_and_stays_within_each_query() -> None:
    frame = pd.DataFrame(
        {
            "query_id": ["q1", "q2", "q1", "q2", "q1", "q2"],
            "score": [10, 20, 11, 21, 12, 22],
        },
        index=[101, 205, 309, 402, 518, 611],
    )

    shuffled = candidate_shuffle_within_query(frame, "score", seed=123)

    assert shuffled.index.equals(frame.index)
    assert shuffled.tolist() == [10, 21, 12, 20, 11, 22]
    assert shuffled.equals(candidate_shuffle_within_query(frame, "score", seed=123))
    for query_id, rows in frame.groupby("query_id", sort=False):
        assert sorted(shuffled.loc[rows.index]) == sorted(
            frame.loc[frame["query_id"] == query_id, "score"]
        )


def test_group_fold_map_is_seeded_order_independent_and_balanced() -> None:
    group_ids = ["g3", "g1", "g2", "g5", "g4", "g3", "g1"]

    mapping = make_group_fold_map(group_ids, folds=3, seed=17)
    reordered = make_group_fold_map(reversed(group_ids), folds=3, seed=17)

    assert mapping == reordered
    assert set(mapping) == {"g1", "g2", "g3", "g4", "g5"}
    assert set(mapping.values()) <= {0, 1, 2}
    fold_sizes = np.bincount(list(mapping.values()), minlength=3)
    assert fold_sizes.max() - fold_sizes.min() <= 1


@pytest.mark.parametrize("folds", [1, 4])
def test_group_fold_map_rejects_invalid_fold_count(folds: int) -> None:
    with pytest.raises(ValueError, match=r"folds must be in \[2, 3\]"):
        make_group_fold_map(["g1", "g2", "g3"], folds=folds, seed=0)


def test_clean_identifier_values_filters_sentinels_and_preserves_first_seen() -> None:
    values = np.asarray(
        [None, "", "  ", "nan", " NaN ", "NONE", "image-1", "image-1", 7],
        dtype=object,
    )

    assert clean_identifier_values(values) == ("image-1", "7")
    assert clean_identifier_values(("image-2", " image-2 ", "image-2")) == (
        "image-2",
        " image-2 ",
    )
    assert clean_identifier_values(None) == ()
    assert clean_identifier_values("image-1") == ()


def test_official_metrics_keep_raw_duplicate_and_none_denominators() -> None:
    retrieved = ["positive-a", "miss", "positive-b", "negative-a", "negative-b"]
    raw_positives = ["positive-a", "positive-a", None, "positive-b"]
    raw_negatives = ["negative-a", "negative-a", None, "negative-b"]

    # The official evaluator makes a set only for hit lookup, but divides by
    # the original four-element annotations, including the duplicate and None.
    assert official_ap_at_k(retrieved, raw_positives, 10) == pytest.approx(5.0 / 12.0)
    assert official_negative_recall_at_k(retrieved, raw_negatives, 10) == pytest.approx(
        0.5
    )


def test_rerank_native_prefix_requires_complete_finite_scores_and_keeps_top10_set() -> (
    None
):
    signatures = [f"candidate-{index}" for index in range(12)]
    artifact = synthetic_model_artifact(
        "model",
        signatures,
        ["00001", "00002", "00003"],
        [
            list(range(12)),
            list(reversed(range(12))),
            [1, 3, 5, 7, 9, 11, 0, 2, 4, 6, 8, 10],
        ],
    )
    original = artifact.top_indices.copy()
    complete = {
        candidate_id: float(index) for index, candidate_id in enumerate(signatures[:10])
    }
    incomplete_ids = artifact.signatures[original[1, :9]].astype(str).tolist()
    incomplete = {
        candidate_id: float(index) for index, candidate_id in enumerate(incomplete_ids)
    }
    nonfinite_ids = artifact.signatures[original[2, :10]].astype(str).tolist()
    nonfinite = {
        candidate_id: (np.nan if index == 4 else float(index))
        for index, candidate_id in enumerate(nonfinite_ids)
    }

    rankings, eligible, changed = rerank_native_prefix(
        artifact,
        {
            "00001": complete,
            "00002": incomplete,
            "00003": nonfinite,
        },
        depth=10,
    )

    assert eligible == {"00001"}
    assert changed == {"00001"}
    assert rankings[0, :10].tolist() == list(reversed(range(10)))
    assert set(rankings[0, :10]) == set(original[0, :10])
    assert np.array_equal(rankings[0, 10:], original[0, 10:])
    assert np.array_equal(rankings[1:], original[1:])
    assert np.array_equal(artifact.top_indices, original)


def test_shared_union_pool_is_one_fixed_identifier_union_for_both_models() -> None:
    first = synthetic_model_artifact(
        "first",
        ["a", "b", "c", "d", "e"],
        ["00001", "00002"],
        [[0, 1, 2], [4, 3, 2]],
    )
    second = synthetic_model_artifact(
        "second",
        ["e", "d", "c", "b", "a"],
        ["00001", "00002"],
        [[2, 1, 4], [0, 3, 2]],
    )

    pools, fingerprint = build_shared_union_pool(first, second, ["00001", "00002"])
    repeated_pools, repeated_fingerprint = build_shared_union_pool(
        first, second, ["00001", "00002"]
    )

    assert pools == {
        "00001": ("a", "b", "c", "d"),
        "00002": ("e", "d", "c", "b"),
    }
    assert repeated_pools == pools
    assert repeated_fingerprint == fingerprint
    for row, query_id in enumerate(("00001", "00002")):
        first_native = set(first.signatures[first.top_indices[row]].astype(str))
        second_native = set(second.signatures[second.top_indices[row]].astype(str))
        assert set(pools[query_id]) == first_native | second_native


def test_cross_model_rerank_uses_reranker_positive_query_and_candidate_space() -> None:
    retriever = synthetic_model_artifact(
        "retriever", ["a", "b", "c"], ["00001"], [[0, 1, 2]]
    )
    reranker = synthetic_model_artifact(
        "reranker", ["c", "a", "b"], ["00001"], [[0, 1, 2]]
    )
    reranker.corpus = np.asarray([[0.2, 0.0], [0.1, 0.0], [0.9, 0.0]], dtype=np.float32)
    reranker.positive_queries = np.asarray([[1.0, 0.0]], dtype=np.float32)

    rankings, changed = cross_model_positive_query_rankings(
        retriever,
        reranker,
        query_ids=["00001"],
        depth=3,
    )

    assert retriever.signatures[rankings[0]].tolist() == ["b", "c", "a"]
    assert changed == {"00001"}


def test_query_balanced_weights_equalize_groups_queries_and_classes() -> None:
    frame = pd.DataFrame(
        {
            "group_id": ["g1"] * 7 + ["g2"] * 3,
            "query_id": ["q1"] * 3 + ["q2"] * 4 + ["q3"] * 3,
            "label_clean": [0, 0, 1, 0, 1, 1, 1, 0, 1, 1],
        },
        index=[11, 13, 17, 19, 23, 29, 31, 37, 41, 43],
    )
    weighted = frame.assign(weight=query_balanced_class_weights(frame))

    group_totals = weighted.groupby("group_id")["weight"].sum()
    assert group_totals.to_dict() == pytest.approx({"g1": 1.0, "g2": 1.0})
    query_totals = weighted.groupby(["group_id", "query_id"])["weight"].sum()
    assert query_totals.to_dict() == pytest.approx(
        {("g1", "q1"): 0.5, ("g1", "q2"): 0.5, ("g2", "q3"): 1.0}
    )
    class_totals = weighted.groupby(["query_id", "label_clean"])["weight"].sum()
    for query_id, query_total in weighted.groupby("query_id")["weight"].sum().items():
        assert class_totals.loc[(query_id, 0)] == pytest.approx(query_total / 2.0)
        assert class_totals.loc[(query_id, 1)] == pytest.approx(query_total / 2.0)


def test_default_features_align_strict_controls_and_expose_sparse_6d_ablations() -> (
    None
):
    specs = {spec.name: spec for spec in default_feature_specs()}

    strict_treatment_controls = (
        ("source_plus_d_available", "source_similarity_strict"),
        ("response_moments", "source_similarity_strict"),
        ("ensemble_plus_d_group", "positive_ensemble_strict"),
        ("source_paraphrase_and_d", "source_plus_paraphrase_strict"),
    )
    for treatment, control in strict_treatment_controls:
        assert specs[treatment].cohort == specs[control].cohort == "strict_any"

    assert specs["imputed_6d_values_only"].columns == RESPONSE_COLUMNS
    assert specs["imputed_6d_values_only"].impute
    assert specs["mask_only"].columns == MASK_COLUMNS
    assert specs["masked_6d"].columns == (*RESPONSE_COLUMNS, *MASK_COLUMNS)
    assert specs["masked_6d"].impute

    permuted_value_columns = specs["category_permuted_6d"].columns[
        : len(PERMUTED_RESPONSE_COLUMNS)
    ]
    assert permuted_value_columns == PERMUTED_RESPONSE_COLUMNS
    assert len(permuted_value_columns) == 6
    assert set(permuted_value_columns).isdisjoint(RESPONSE_COLUMNS)
    assert specs["category_permuted_6d"].columns == (
        *PERMUTED_RESPONSE_COLUMNS,
        *PERMUTED_MASK_COLUMNS,
    )


def test_ranking_set_jaccard_reports_set_and_overlap_fraction() -> None:
    overlap, jaccard, overlap_fraction, exact = ranking_set_jaccard(
        ["a", "b", "c"], ["a", "c", "d"], 3
    )

    assert overlap == 2
    assert jaccard == pytest.approx(0.5)
    assert overlap_fraction == pytest.approx(2 / 3)
    assert not exact

    assert ranking_set_jaccard(["a", "b"], ["b", "a"], 2) == (2, 1.0, 1.0, True)
    with pytest.raises(ValueError, match="duplicate"):
        ranking_set_jaccard(["a", "a"], ["a", "b"], 2)


def test_simple_topk_recall_and_ndcg_metrics() -> None:
    retrieved = ["p2", "x", "p1", "y"]
    positives = ["p1", "p2", "p3"]

    assert positive_recall_at_k(retrieved, positives, 2) == pytest.approx(0.5)
    assert positive_recall_at_k(retrieved, positives, 4) == pytest.approx(2 / 3)
    expected_dcg = 1.0 + 1.0 / np.log2(4)
    expected_idcg = 1.0 + 1.0 / np.log2(3) + 1.0 / np.log2(4)
    assert binary_ndcg_at_k(retrieved, positives, 4) == pytest.approx(
        expected_dcg / expected_idcg
    )


def _query_metric_frame(
    query_ids: list[str], ap_values: list[float], auc_values: list[float]
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "group_id": [f"g{index}" for index in range(len(query_ids))],
            "query_id": query_ids,
            "candidate_cohort_fingerprint": [
                f"cohort::{query_id}" for query_id in query_ids
            ],
            "query_ap": ap_values,
            "query_pairwise_auc": auc_values,
        }
    )


def test_cross_model_probe_comparison_skips_model_dependent_rank_cohort() -> None:
    fixed_baseline = _query_metric_frame(["q1", "q2"], [0.2, 0.4], [0.3, 0.5])
    fixed_hard = _query_metric_frame(["q1", "q2"], [0.3, 0.5], [0.4, 0.6])
    baseline = {
        ("native_clean", "source_similarity_strict"): fixed_baseline,
        (
            "rank_violating_official_negative",
            "source_similarity_strict",
        ): _query_metric_frame(["q1", "q2"], [0.1, 0.2], [0.2, 0.3]),
    }
    hard = {
        ("native_clean", "source_similarity_strict"): fixed_hard,
        (
            "rank_violating_official_negative",
            "source_similarity_strict",
        ): _query_metric_frame(["q1"], [0.2], [0.3]),
    }

    result = paired_model_probe_statistics(
        baseline,
        hard,
        baseline_name="baseline",
        hard_name="hard",
        pool_name="shared",
        bootstrap_replicates=0,
        seed=7,
    )

    assert set(result["scope"]) == {"native_clean"}
    assert result["estimate"].tolist() == pytest.approx([0.1, 0.1])


def test_cross_model_probe_comparison_rejects_candidate_cohort_mismatch() -> None:
    baseline_frame = _query_metric_frame(["q1"], [0.2], [0.3])
    hard_frame = _query_metric_frame(["q1"], [0.3], [0.4])
    hard_frame.loc[0, "candidate_cohort_fingerprint"] = "different-candidates"

    with pytest.raises(ValueError, match="candidate/label cohort differs"):
        paired_model_probe_statistics(
            {("native_clean", "source_similarity_strict"): baseline_frame},
            {("native_clean", "source_similarity_strict"): hard_frame},
            baseline_name="baseline",
            hard_name="hard",
            pool_name="shared",
            bootstrap_replicates=0,
            seed=7,
        )


def test_incremental_probe_did_is_paired_on_the_shared_fixed_cohort() -> None:
    keys = {
        "control": ("pool_nonpositive_benchmark", "source_similarity_strict"),
        "enhanced": ("pool_nonpositive_benchmark", "source_plus_d_available"),
    }
    baseline = {
        keys["control"]: _query_metric_frame(["q1", "q2"], [0.20, 0.40], [0.30, 0.50]),
        keys["enhanced"]: _query_metric_frame(["q1", "q2"], [0.25, 0.45], [0.35, 0.55]),
    }
    hard = {
        keys["control"]: _query_metric_frame(["q1", "q2"], [0.30, 0.50], [0.40, 0.60]),
        keys["enhanced"]: _query_metric_frame(["q1", "q2"], [0.50, 0.70], [0.60, 0.80]),
    }

    result = paired_model_incremental_probe_did(
        baseline,
        hard,
        baseline_name="baseline",
        hard_name="hard",
        pool_name="shared",
        bootstrap_replicates=0,
        seed=11,
    )

    assert len(result) == 2
    assert set(result["metric"]) == {"query_ap", "query_pairwise_auc"}
    assert result["estimate"].tolist() == pytest.approx([0.15, 0.15])
    assert set(result["query_count"]) == {2}
