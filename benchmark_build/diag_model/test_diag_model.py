from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import CELL_LABELS, EDIT_OPERATIONS, SEMANTIC_TARGETS  # noqa: E402
from analyze_delta_modes import wilson_interval  # noqa: E402
from extract_embeddings import (  # noqa: E402
    Negative,
    Sample,
    assemble_outputs,
    cache_path,
    encode_reference_query_batch,
    reference_cache_key,
)
from probe_grid import (  # noqa: E402
    factorial_structure,
    make_feature_views,
    reference_ablation_cell_rows,
    run_numeric_probe,
)
from paraphrase_robustness import (  # noqa: E402
    aggregate_robustness,
    classify_intervals,
    pair_balanced_interval,
    pair_equal_weights,
    weighted_quantile,
)
from pinpoint_full_diagnostics import (  # noqa: E402
    group_balanced_quantile,
    hit_at_k,
    ndcg_at_k,
    recall_at_k,
    topk_jaccard,
)
from qwen3vl_adapter import (  # noqa: E402
    QUERY_INSTRUCTION,
    TARGET_INSTRUCTION,
    build_query_inputs,
    build_reference_query_inputs,
    build_target_inputs,
    protocol_metadata,
    reference_protocol_metadata,
)


def synthetic_grid(seed: int = 7):
    rng = np.random.default_rng(seed)
    dimension = 16
    semantic_vectors = rng.normal(size=(3, dimension))
    operation_vectors = rng.normal(size=(3, dimension))
    semantic_vectors /= np.linalg.norm(semantic_vectors, axis=1, keepdims=True)
    operation_vectors /= np.linalg.norm(operation_vectors, axis=1, keepdims=True)

    vectors = []
    semantic = []
    operation = []
    cells = []
    groups = []
    for group_index in range(18):
        group_noise = rng.normal(scale=0.01, size=dimension)
        for semantic_index, semantic_name in enumerate(SEMANTIC_TARGETS):
            for operation_index, operation_name in enumerate(EDIT_OPERATIONS):
                vectors.append(
                    semantic_vectors[semantic_index]
                    + operation_vectors[operation_index]
                    + group_noise
                    + rng.normal(scale=0.01, size=dimension)
                )
                semantic.append(semantic_name)
                operation.append(operation_name)
                cells.append(f"{semantic_name}|{operation_name}")
                groups.append(str(group_index))
    labels = {
        "semantic_target": np.asarray(semantic, dtype=object),
        "edit_operation": np.asarray(operation, dtype=object),
        "cell": np.asarray(cells, dtype=object),
        "group": np.asarray(groups, dtype=object),
    }
    return np.asarray(vectors, dtype=np.float32), labels


def test_delta_identity() -> None:
    rng = np.random.default_rng(3)
    q_positive = rng.normal(size=(5, 8)).astype(np.float32)
    q_positive /= np.linalg.norm(q_positive, axis=1, keepdims=True)
    target = rng.normal(size=(5, 8)).astype(np.float32)
    target /= np.linalg.norm(target, axis=1, keepdims=True)
    sample_index = np.repeat(np.arange(5), 3)
    q_negative = rng.normal(size=(15, 8)).astype(np.float32)
    q_negative /= np.linalg.norm(q_negative, axis=1, keepdims=True)
    _, scores = make_feature_views(
        {
            "q_positive": q_positive,
            "target": target,
            "q_negative": q_negative,
            "sample_index": sample_index,
        }
    )
    assert float(scores["identity_error"].max()) < 1e-6


def test_wilson_interval_is_non_degenerate_for_sparse_all_wins() -> None:
    low, high = wilson_interval(7, 7)
    assert 0.5 < low < 1.0
    assert np.isclose(high, 1.0)


def test_pair_balanced_paraphrase_interval_does_not_overweight_dense_pairs() -> None:
    values = np.asarray([-0.2, -0.1, 0.4], dtype=np.float64)
    groups = np.asarray(["a", "a", "b"], dtype=object)
    weights = pair_equal_weights(groups)
    assert np.allclose(weights, [0.5, 0.5, 1.0])
    assert weighted_quantile(values, weights, 0.5) == -0.1

    duplicated = np.asarray([-0.2, -0.2, -0.1, -0.1, 0.4], dtype=np.float64)
    duplicated_groups = np.asarray(["a", "a", "a", "a", "b"], dtype=object)
    assert pair_balanced_interval(values, groups, 0.5) == pair_balanced_interval(
        duplicated, duplicated_groups, 0.5
    )


def test_paraphrase_interval_classification_has_uncertain_band() -> None:
    delta = np.asarray([0.03, 0.005, -0.04])
    low = np.full(3, -0.01)
    high = np.full(3, 0.02)
    lower, upper, state = classify_intervals(delta, low, high)
    assert np.allclose(lower, [0.02, -0.005, -0.05])
    assert np.allclose(upper, [0.05, 0.025, -0.02])
    assert state.tolist() == ["stable_win", "uncertain", "stable_loss"]


def test_observed_variant_win_rate_is_pair_balanced() -> None:
    frame = pd.DataFrame(
        {
            "pair_id": ["a", "a", "b"],
            "edit_operation": ["add", "add", "add"],
            "construction_mode": ["text_only_editable"] * 3,
            "semantic_target": ["object"] * 3,
            "delta": [0.1, -0.1, 0.05],
            "original_win": [1.0, 0.0, 1.0],
            "calibrated_state": ["stable_win", "stable_loss", "uncertain"],
            "calibrated_delta_low": [0.01, -0.2, -0.01],
            "calibrated_delta_high": [0.2, -0.01, 0.1],
            "observed_state": ["stable_win", "stable_loss", "uncertain"],
            "observed_win_fraction": [1.0, 0.0, 0.25],
            "observed_delta_mean": [0.1, -0.1, 0.01],
            "observed_delta_min": [0.05, -0.2, -0.02],
        }
    )
    summary = aggregate_robustness(
        frame,
        bootstrap_replicates=100,
        seed=42,
    )
    row = summary.loc[summary["subset"] == "strict"].iloc[0]
    assert np.isclose(row["pair_balanced_observed_variant_win_rate"], 0.375)
    assert np.isclose(row["pair_balanced_observed_mean_delta"], 0.005)
    assert np.isclose(row["pair_balanced_observed_worst_delta"], -0.0475)


def test_pinpoint_ndcg_uses_binary_positive_relevance() -> None:
    positives = {3, 7}
    assert np.isclose(ndcg_at_k([3, 7, 1], positives, 2), 1.0)
    expected = (1.0 / np.log2(3.0)) / (
        1.0 + 1.0 / np.log2(3.0)
    )
    assert np.isclose(ndcg_at_k([1, 3, 7], positives, 2), expected)
    assert ndcg_at_k([1, 2, 4], positives, 3) == 0.0


def test_pinpoint_recall_and_hit_have_explicit_set_semantics() -> None:
    positives = {3, 7, 9, 11}
    assert np.isclose(recall_at_k([3, 7, 1], positives, 3), 0.5)
    assert np.isclose(recall_at_k([3, 7, 9, 11], positives, 2), 0.5)
    assert hit_at_k([1, 7, 9], positives, 1) == 0.0
    assert hit_at_k([1, 7, 9], positives, 2) == 1.0


def test_pinpoint_topk_jaccard_uses_retrieval_sets() -> None:
    assert topk_jaccard([1, 2, 3], [2, 3, 4], 3) == 0.5
    assert topk_jaccard([1, 2], [1, 3], 1) == 1.0


def test_pinpoint_paraphrase_threshold_balances_groups() -> None:
    values = {
        "dense": np.asarray([0.1, 0.2, 0.3, 0.4]),
        "sparse": np.asarray([0.9]),
    }
    duplicated = {
        "dense": np.repeat(values["dense"], 5),
        "sparse": values["sparse"],
    }
    assert group_balanced_quantile(values, 0.5) == group_balanced_quantile(
        duplicated,
        0.5,
    )


def test_qwen_payloads_keep_composed_query_and_candidate_roles() -> None:
    query = build_query_inputs(
        [
            {
                "query_image_signature": "/images/reference.png",
                "instruction": "make the dog black",
            }
        ]
    )
    target = build_target_inputs([Path("/images/target.png")])
    reference = build_reference_query_inputs([Path("/images/reference.png")])

    assert query == [
        {
            "image": "/images/reference.png",
            "text": "make the dog black",
            "instruction": QUERY_INSTRUCTION,
        }
    ]
    assert target == [
        {
            "image": "/images/target.png",
            "instruction": TARGET_INSTRUCTION,
        }
    ]
    assert reference == [
        {
            "image": "/images/reference.png",
            "instruction": QUERY_INSTRUCTION,
        }
    ]
    metadata = protocol_metadata()
    assert metadata["pooling"] == "last_non_padding_token"
    assert metadata["query_content"] == ["reference_image", "modification_text"]
    reference_metadata = reference_protocol_metadata()
    assert reference_metadata["content"] == ["reference_image"]
    assert reference_metadata["instruction"] == QUERY_INSTRUCTION


def test_reference_baseline_uses_model_specific_query_role() -> None:
    class FakeAdapter:
        def encode_image_batch(self, paths):
            return ("base", paths)

        def _encode(self, paths, texts, q_or_c):
            return ("mllm", paths, texts, q_or_c)

        def encode_reference_query_batch(self, paths):
            return ("qwen", paths)

    adapter = FakeAdapter()
    paths = [Path("/images/reference.png")]
    assert encode_reference_query_batch(adapter, "BGE-VL-base", paths) == (
        "base",
        paths,
    )
    assert encode_reference_query_batch(adapter, "BGE-VL-MLLM-S1", paths) == (
        "mllm",
        paths,
        None,
        "q",
    )
    assert encode_reference_query_batch(
        adapter, "Qwen3-VL-Embedding-2B", paths
    ) == ("qwen", paths)


def test_reference_contribution_identity() -> None:
    rng = np.random.default_rng(13)
    sample_count = 6
    dimension = 8
    q_reference = rng.normal(size=(sample_count, dimension)).astype(np.float32)
    q_positive = rng.normal(size=(sample_count, dimension)).astype(np.float32)
    target = rng.normal(size=(sample_count, dimension)).astype(np.float32)
    for value in (q_reference, q_positive, target):
        value /= np.linalg.norm(value, axis=1, keepdims=True)
    sample_index = np.repeat(np.arange(sample_count), 3)
    q_negative = rng.normal(size=(len(sample_index), dimension)).astype(np.float32)
    q_negative /= np.linalg.norm(q_negative, axis=1, keepdims=True)
    _, scores = make_feature_views(
        {
            "q_reference": q_reference,
            "q_positive": q_positive,
            "target": target,
            "q_negative": q_negative,
            "sample_index": sample_index,
        }
    )
    assert float(scores["reference_contribution_identity_error"].max()) < 1e-6
    assert np.allclose(
        scores["delta"],
        scores["positive_gain_from_reference"]
        - scores["negative_gain_from_reference"],
        atol=1e-6,
    )
    assert np.max(np.abs(scores["positive_reference_local_alignment"])) <= 1.0 + 1e-6

    labels = {
        "semantic_target": np.resize(np.asarray(SEMANTIC_TARGETS, dtype=object), 18),
        "edit_operation": np.tile(np.asarray(EDIT_OPERATIONS, dtype=object), 6),
        "construction_mode": np.resize(
            np.asarray(["text_only_editable", "vision_grounded"], dtype=object),
            18,
        ),
        "group": np.asarray([str(index) for index in sample_index], dtype=object),
    }
    rows = reference_ablation_cell_rows(scores, labels, 0, 42)
    assert len(rows) == 27
    assert any(row["count"] > 0 for row in rows)


def test_extraction_assembly_includes_reference_contributions(tmp_path: Path) -> None:
    samples = [
        Sample(0, "s0", "p0", "/images/r0.png", "/images/t0.png", "make red"),
        Sample(1, "s1", "p1", "/images/r1.png", "/images/t1.png", "add dog"),
    ]
    negatives = [
        Negative(
            0,
            0,
            "s0",
            "p0",
            "n0",
            "make red",
            "make blue",
            "attribute",
            "substitute",
            "text_only_editable",
        ),
        Negative(
            1,
            1,
            "s1",
            "p1",
            "n1",
            "add dog",
            "add cat",
            "object",
            "substitute",
            "vision_grounded",
        ),
    ]
    vectors = {
        "r0": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        "r1": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        "p0": np.asarray([0.8, 0.6, 0.0], dtype=np.float32),
        "p1": np.asarray([0.0, 0.8, 0.6], dtype=np.float32),
        "n0": np.asarray([0.6, 0.8, 0.0], dtype=np.float32),
        "n1": np.asarray([0.0, 0.6, 0.8], dtype=np.float32),
        "t0": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        "t1": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
    }
    output_dir = tmp_path / "features"
    cache_root = output_dir / "cache"

    def save(role: str, logical_key: str, vector_key: str) -> None:
        path = cache_path(cache_root, role, logical_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vectors[vector_key], allow_pickle=False)

    for sample in samples:
        save("query", f"positive:{sample.sample_id}", sample.pair_id)
        save("target", sample.sample_id, f"t{sample.sample_index}")
        save(
            "reference",
            reference_cache_key(sample.reference_path),
            f"r{sample.sample_index}",
        )
    for negative in negatives:
        save("query", f"negative:{negative.negative_id}", negative.negative_id)

    metadata = assemble_outputs(
        output_dir,
        cache_root,
        samples,
        negatives,
        {"model_name": "synthetic"},
    )
    with np.load(output_dir / "features.npz") as archive:
        assert archive["q_reference"].shape == (2, 3)
    header = (output_dir / "scores.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "reference_score" in header
    assert "positive_gain_from_reference" in header
    assert metadata["reference_contribution_identity_max_abs_error"] < 1e-6


def test_grouped_linear_probe_recovers_grid() -> None:
    vectors, labels = synthetic_grid()
    report, predictions = run_numeric_probe(
        "synthetic",
        vectors,
        labels["cell"],
        labels["group"],
        CELL_LABELS,
        folds=3,
        inner_folds=2,
        c_grid=(1.0,),
        tune_c=True,
        standardize=False,
        bootstrap_replicates=20,
        seed=42,
        max_iterations=1000,
    )
    assert predictions is not None
    assert report["status"] == "ok"
    assert report["metrics"]["accuracy"]["value"] > 0.95
    assert report["group_count"] == 18


def test_additive_factor_model_matches_saturated_model() -> None:
    vectors, labels = synthetic_grid()
    report = factorial_structure(vectors, labels, folds=3, seed=42, alpha=1e-3)
    assert report["status"] == "ok"
    assert report["additive"]["frobenius_r2_against_fold_train_mean"] > 0.95
    assert abs(report["interaction_gain_r2"]) < 0.02


def test_probe_cli_writes_complete_report(tmp_path: Path) -> None:
    vectors, labels = synthetic_grid()
    group_values = sorted(set(labels["group"].tolist()), key=int)
    group_to_index = {value: index for index, value in enumerate(group_values)}
    sample_index = np.asarray([group_to_index[value] for value in labels["group"]])
    dimension = vectors.shape[1]
    q_positive = np.zeros((len(group_values), dimension), dtype=np.float32)
    q_positive[:, 0] = 1.0
    target = q_positive.copy()
    q_reference = np.roll(q_positive, shift=1, axis=1)
    q_negative = q_positive[sample_index] - 0.05 * vectors
    q_negative /= np.linalg.norm(q_negative, axis=1, keepdims=True)

    features_dir = tmp_path / "features"
    output_dir = tmp_path / "analysis"
    features_dir.mkdir()
    np.savez(
        features_dir / "features.npz",
        q_reference=q_reference,
        q_positive=q_positive,
        target=target,
        q_negative=q_negative,
        sample_index=sample_index,
    )
    (features_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model_name": "synthetic",
                "sample_count": len(group_values),
                "negative_count": len(vectors),
                "embedding_dimension": dimension,
            }
        ),
        encoding="utf-8",
    )
    with (features_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for index, (semantic, operation, group) in enumerate(
            zip(labels["semantic_target"], labels["edit_operation"], labels["group"])
        ):
            handle.write(
                json.dumps(
                    {
                        "negative_index": index,
                        "sample_index": int(sample_index[index]),
                        "sample_id": f"sample-{group}",
                        "pair_id": group,
                        "negative_id": f"negative-{index}",
                        "positive_text": "positive text",
                        "negative_text": f"negative {semantic} {operation}",
                        "semantic_target": semantic,
                        "edit_operation": operation,
                        "construction_mode": (
                            "text_only_editable" if index % 2 == 0 else "vision_grounded"
                        ),
                    }
                )
                + "\n"
            )

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / "probe_grid.py"),
            "--features-dir",
            str(features_dir),
            "--output-dir",
            str(output_dir),
            "--folds",
            "3",
            "--inner-folds",
            "2",
            "--c-grid",
            "1.0",
            "--bootstrap-replicates",
            "5",
        ],
        check=True,
    )
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert report["analysis"]["grouped_by"] == "pair_id"
    assert report["analysis"]["reference_ablation_available"] is True
    assert (output_dir / "delta_by_cell.csv").is_file()
    assert (output_dir / "reference_ablation_by_cell.csv").is_file()
    assert (output_dir / "probe_metrics.csv").is_file()
    assert (output_dir / "summary.md").is_file()

    descriptive_dir = tmp_path / "descriptive_analysis"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / "probe_grid.py"),
            "--features-dir",
            str(features_dir),
            "--output-dir",
            str(descriptive_dir),
            "--bootstrap-replicates",
            "5",
            "--descriptive-only",
        ],
        check=True,
    )
    descriptive_report = json.loads(
        (descriptive_dir / "descriptive_report.json").read_text(encoding="utf-8")
    )
    assert descriptive_report["analysis"]["descriptive_only"] is True
    assert (descriptive_dir / "delta_by_cell.csv").is_file()
    assert (descriptive_dir / "reference_ablation_by_cell.csv").is_file()
    assert not (descriptive_dir / "probe_metrics.csv").exists()

    visualization_dir = tmp_path / "vector_visualization"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / "visualize_vectors.py"),
            "--features-dir",
            str(features_dir),
            "--output-dir",
            str(visualization_dir),
            "--pca-dim",
            "10",
            "--perplexity",
            "10",
            "--tsne-iterations",
            "250",
            "--neighbors",
            "5",
            "--silhouette-sample-size",
            "100",
            "--dpi",
            "60",
        ],
        check=True,
    )
    visualization_report = json.loads(
        (visualization_dir / "report.json").read_text(encoding="utf-8")
    )
    assert visualization_report["analyzed_row_count"] == len(vectors)
    assert set(visualization_report["views"]) == {"raw_v", "direction_v"}
    assert (visualization_dir / "coordinates.csv").is_file()
    assert (visualization_dir / "quality_metrics.csv").is_file()
    assert (visualization_dir / "pca_raw_v.png").is_file()
    assert (visualization_dir / "tsne_direction_v.png").is_file()
