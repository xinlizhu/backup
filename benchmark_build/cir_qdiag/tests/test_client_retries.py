from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from cir_qdiag.models.client import StructuredVLMClient
from cir_qdiag.models.fake import FakeVLMBackend


class TinyResponse(BaseModel):
    value: str


class HTTPFailure(RuntimeError):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


def _call(client: StructuredVLMClient):
    return client.call(
        TinyResponse,
        images=[],
        system_prompt="system",
        user_prompt='{"input":"x"}',
        stage="tiny",
        temperature=0.0,
        seed=10,
        max_new_tokens=16,
    )


def test_non_retryable_400_stops_after_one_attempt(tmp_path: Path) -> None:
    backend = FakeVLMBackend([HTTPFailure(400, "input too long"), {"value": "unused"}])
    client = StructuredVLMClient(backend, tmp_path, max_retries=2)

    with pytest.raises(RuntimeError, match="input too long"):
        _call(client)

    assert len(backend.calls) == 1
    error = json.loads(next(tmp_path.glob("tiny/*.error.json")).read_text(encoding="utf-8"))
    assert error["error_details"][0]["retryable"] is False


def test_validation_error_retries_and_preserves_backend_response(tmp_path: Path) -> None:
    invalid_response = {"unexpected": "value"}
    backend = FakeVLMBackend([invalid_response, {"value": "recovered"}])
    client = StructuredVLMClient(backend, tmp_path, max_retries=2)

    response, trace = _call(client)

    assert response.value == "recovered"
    assert trace.attempts == 2
    assert len(backend.calls) == 2
    cache = json.loads(next(tmp_path.glob("tiny/*.json")).read_text(encoding="utf-8"))
    assert len(cache["previous_error_details"]) == 1
    detail = cache["previous_error_details"][0]
    assert detail["type"] == "ValidationError"
    assert detail["retryable"] is True
    assert detail["effective_seed"] == 10
    assert detail["response_data"] == invalid_response
    assert detail["raw_response"] == {"fake": True, "data": invalid_response}
    assert detail["request_id"] == "fake-1"


def test_server_error_retries_with_incremented_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("cir_qdiag.models.client.time.sleep", lambda _: None)
    backend = FakeVLMBackend([HTTPFailure(500, "temporary"), {"value": "ok"}])
    client = StructuredVLMClient(backend, tmp_path, max_retries=2)

    response, trace = _call(client)

    assert response.value == "ok"
    assert trace.attempts == 2
    assert trace.seed == 10
    assert trace.effective_seed == 11
    assert [call["seed"] for call in backend.calls] == [10, 11]
    cache = json.loads(next(tmp_path.glob("tiny/*.json")).read_text(encoding="utf-8"))
    assert cache["previous_error_details"] == [
        {
            "attempt": 1,
            "type": "HTTPFailure",
            "message": "temporary",
            "retryable": True,
            "effective_seed": 10,
        }
    ]


def test_backend_identity_is_part_of_cache_key(tmp_path: Path) -> None:
    first_backend = FakeVLMBackend(
        [{"value": "first"}], identity={"backend_contract": "fake", "max_pixels": 1}
    )
    first, first_trace = _call(StructuredVLMClient(first_backend, tmp_path, max_retries=0))
    second_backend = FakeVLMBackend(
        [{"value": "second"}], identity={"backend_contract": "fake", "max_pixels": 2}
    )
    second, second_trace = _call(StructuredVLMClient(second_backend, tmp_path, max_retries=0))

    assert first.value == "first"
    assert second.value == "second"
    assert first_trace.input_hash != second_trace.input_hash
    assert len(second_backend.calls) == 1


def test_semantically_rejected_cache_is_quarantined_and_regenerated(tmp_path: Path) -> None:
    backend = FakeVLMBackend([{"value": "bad-domain-value"}, {"value": "corrected"}])
    client = StructuredVLMClient(backend, tmp_path, max_retries=0)

    first, first_trace = _call(client)
    rejected_path = client.reject_cached_response(first_trace, "domain check failed")
    second, second_trace = _call(client)

    assert first.value == "bad-domain-value"
    assert second.value == "corrected"
    assert rejected_path is not None and rejected_path.is_file()
    rejected = json.loads(rejected_path.read_text(encoding="utf-8"))
    assert rejected["semantic_rejection"]["reason"] == "domain check failed"
    assert second_trace.cached is False
    assert len(backend.calls) == 2
