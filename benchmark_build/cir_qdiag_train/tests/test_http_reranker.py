from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cir_qdiag_train.http_reranker import (
    HttpRerankerClient,
    _detect_image_mime,
    _parse_rerank_scores,
    _parse_score_scores,
)
from cir_qdiag_train.on_policy_config import OnPolicyTeacherConfig
from cir_qdiag_train.on_policy_mining import _score_http_reranker_candidates


def test_parse_reranker_scores_restores_original_candidate_order() -> None:
    response = {
        "results": [
            {"index": 2, "relevance_score": 0.1},
            {"index": 0, "relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.4},
        ]
    }
    scores = _parse_rerank_scores(response, expected_count=3, pair_id=7)
    assert scores.dtype == np.float32
    assert np.allclose(scores, [0.9, 0.4, 0.1])


def test_parse_score_endpoint_scores_restores_original_candidate_order() -> None:
    response = {
        "data": [
            {"index": 1, "object": "score", "score": 0.2},
            {"index": 0, "object": "score", "score": 0.8},
        ]
    }
    scores = _parse_score_scores(response, expected_count=2, pair_id=9)
    assert np.allclose(scores, [0.8, 0.2])


def test_http_reranker_payload_is_multimodal_and_omits_instruction(
    tmp_path: Path,
) -> None:
    image = tmp_path / "sample.jpg"
    image.write_bytes(b"\xff\xd8\xff" + b"payload")
    config = OnPolicyTeacherConfig(
        backend="http_reranker",
        endpoint_url="http://reranker.test/v1/rerank",
        model_name="qwen3-vl-reranker-2b",
        instruction=None,
        image_transport="data_url",
    )
    client = HttpRerankerClient(config)
    captured = {}

    def fake_post(payload, *, pair_id):
        captured["payload"] = payload
        captured["pair_id"] = pair_id
        return {
            "results": [
                {"index": 1, "relevance_score": 0.2},
                {"index": 0, "relevance_score": 0.8},
            ]
        }

    client._post_json = fake_post  # type: ignore[method-assign]
    record = SimpleNamespace(
        pair_id=5,
        reference_path=image,
        text="make it silver",
    )
    scores = client.score_candidates(
        record=record,
        candidate_paths=[image, image],
    )
    payload = captured["payload"]
    assert "instruction" not in payload
    assert payload["top_n"] == 2
    assert payload["query"]["content"][0]["type"] == "image_url"
    assert payload["query"]["content"][1] == {
        "type": "text",
        "text": "make it silver",
    }
    image_url = payload["documents"][0]["content"][0]["image_url"]["url"]
    assert image_url.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(image_url.split(",", 1)[1]).startswith(b"\xff\xd8\xff")
    assert captured["pair_id"] == 5
    assert np.allclose(scores, [0.8, 0.2])


def test_image_mime_uses_magic_bytes_before_misleading_suffix(tmp_path: Path) -> None:
    assert _detect_image_mime(b"\xff\xd8\xffdata", tmp_path / "wrong.png") == (
        "image/jpeg"
    )


def test_http_score_cache_is_reused_after_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    teacher = OnPolicyTeacherConfig(
        backend="http_reranker",
        endpoint_url="http://reranker.test/v1/rerank",
        model_name="qwen3-vl-reranker-2b",
        max_concurrent_requests=1,
    )
    config = SimpleNamespace(teacher=teacher)
    candidates = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
    records = [
        SimpleNamespace(pair_id=10),
        SimpleNamespace(pair_id=11),
    ]

    class FakeClient:
        def __init__(self, config) -> None:
            assert config is teacher

        def score_candidates(self, *, record, candidate_paths):
            assert len(candidate_paths) == 2
            return np.asarray(
                [0.9, 0.1] if record.pair_id == 10 else [0.4, 0.8],
                dtype=np.float32,
            )

    monkeypatch.setattr(
        "cir_qdiag_train.on_policy_mining.HttpRerankerClient", FakeClient
    )
    scores, top1 = _score_http_reranker_candidates(
        config=config,
        candidate_indices=candidates,
        records=records,
        image_paths=[tmp_path / "i0", tmp_path / "i1"],
        output_dir=tmp_path,
    )
    assert np.allclose(scores, [[0.9, 0.1], [0.4, 0.8]])
    assert top1.tolist() == [True, False]

    class FailIfCalled(FakeClient):
        def score_candidates(self, *, record, candidate_paths):
            raise AssertionError("completed HTTP score cache was not reused")

    monkeypatch.setattr(
        "cir_qdiag_train.on_policy_mining.HttpRerankerClient", FailIfCalled
    )
    repeated, repeated_top1 = _score_http_reranker_candidates(
        config=config,
        candidate_indices=candidates,
        records=records,
        image_paths=[tmp_path / "i0", tmp_path / "i1"],
        output_dir=tmp_path,
    )
    assert np.array_equal(repeated, scores)
    assert np.array_equal(repeated_top1, top1)
