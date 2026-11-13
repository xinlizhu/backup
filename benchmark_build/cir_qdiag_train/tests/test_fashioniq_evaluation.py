from __future__ import annotations

import json

import numpy as np
import torch
from PIL import Image

from cir_qdiag_train.fashioniq_evaluation import (
    aggregate_category_metrics,
    cache_feature_matrix,
    evaluate_feature_files,
    join_fashioniq_captions,
    load_fashioniq_val,
)


def test_join_fashioniq_captions_matches_local_protocol() -> None:
    assert (
        join_fashioniq_captions(["is black.", " has shorter sleeves? "])
        == "is black and has shorter sleeves"
    )
    assert join_fashioniq_captions(["", "is grey"]) == " and is grey"


def test_load_cache_and_evaluate_fashioniq(tmp_path) -> None:
    dataset_root = tmp_path / "fashion-iq"
    (dataset_root / "captions").mkdir(parents=True)
    (dataset_root / "image_splits").mkdir()
    (dataset_root / "images").mkdir()
    gallery_ids = ["a", "b", "c"]
    (dataset_root / "image_splits/split.dress.val.json").write_text(
        json.dumps(gallery_ids), encoding="utf-8"
    )
    (dataset_root / "captions/cap.dress.val.json").write_text(
        json.dumps(
            [
                {
                    "candidate": "a",
                    "target": "b",
                    "captions": ["is blue", "has sleeves"],
                },
                {
                    "candidate": "b",
                    "target": "c",
                    "captions": ["is red", "is long"],
                },
            ]
        ),
        encoding="utf-8",
    )
    for image_id in gallery_ids:
        Image.new("RGB", (2, 2)).save(dataset_root / "images" / f"{image_id}.png")
    loaded_ids, _, records = load_fashioniq_val(dataset_root, "dress")
    assert loaded_ids == gallery_ids
    assert records[0].text == "is blue and has sleeves"

    gallery_path = tmp_path / "gallery.npy"
    query_path = tmp_path / "queries.npy"
    gallery_values = np.eye(3, dtype=np.float32)
    query_values = np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    cache_feature_matrix(
        gallery_path,
        tmp_path / "gallery_progress.json",
        count=3,
        batch_size=2,
        encode=lambda begin, end: gallery_values[begin:end],
        description="test gallery",
    )
    cache_feature_matrix(
        query_path,
        tmp_path / "query_progress.json",
        count=2,
        batch_size=1,
        encode=lambda begin, end: query_values[begin:end],
        description="test queries",
    )
    metrics = evaluate_feature_files(
        gallery_path=gallery_path,
        query_path=query_path,
        gallery_ids=gallery_ids,
        records=records,
        device=torch.device("cpu"),
        score_batch_size=1,
        recall_k=(1, 2),
    )
    assert metrics["recall_at_1"] == 100.0
    assert metrics["recall_at_2"] == 100.0


def test_aggregate_category_metrics_uses_macro_average() -> None:
    categories = {
        category: {
            "num_queries": count,
            "gallery_size": 100,
            "recall_at_1": value,
            "recall_at_5": value,
            "recall_at_10": value,
            "recall_at_50": value,
        }
        for category, count, value in (
            ("dress", 1, 10.0),
            ("shirt", 2, 20.0),
            ("toptee", 3, 30.0),
        )
    }
    result = aggregate_category_metrics(
        model_name="model",
        model_source="checkpoint",
        category_metrics=categories,
    )
    assert result["macro_average"]["recall_at_10"] == 20.0
    assert result["micro_average"]["recall_at_10"] == 23.3333
    assert result["num_queries"] == 6
