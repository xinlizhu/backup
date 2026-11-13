from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from cir_qdiag_openai_cirr.models.base import FatalProviderError
from cir_qdiag_openai_cirr.models.client import StructuredLLMClient
from cir_qdiag_openai_cirr.models.fake import FakeLLMBackend


class TinyResponse(BaseModel):
    value: str


class HTTPFailure(RuntimeError):
    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        retry_after: int | None = None,
        body: object | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        if body is None and retry_after is not None:
            self.body = {"error": {"retry_after": retry_after}}


def _call(client: StructuredLLMClient):
    return client.call(
        TinyResponse,
        images=[],
        system_prompt="system",
        user_prompt='{"input":"x"}',
        stage="tiny",
        max_output_tokens=16,
    )


def test_non_retryable_400_stops_after_one_attempt(tmp_path: Path) -> None:
    backend = FakeLLMBackend([HTTPFailure(400, "input too long"), {"value": "unused"}])
    client = StructuredLLMClient(backend, tmp_path, max_retries=2)

    with pytest.raises(RuntimeError, match="input too long"):
        _call(client)

    assert len(backend.calls) == 1
    error = json.loads(next(tmp_path.glob("tiny/*.error.json")).read_text(encoding="utf-8"))
    assert error["error_details"][0]["retryable"] is False


def test_daily_quota_failure_is_provider_fatal(tmp_path: Path) -> None:
    backend = FakeLLMBackend(
        [
            HTTPFailure(
                403,
                "daily subscription quota exhausted",
                body={
                    "code": "DAILY_LIMIT_EXCEEDED",
                    "message": "您当前每日订阅额度不足，请手动重置额度。",
                },
            ),
            {"value": "unused"},
        ]
    )
    client = StructuredLLMClient(backend, tmp_path, max_retries=2)

    with pytest.raises(FatalProviderError, match="provider account or quota"):
        _call(client)

    assert len(backend.calls) == 1
    error = json.loads(next(tmp_path.glob("tiny/*.error.json")).read_text(encoding="utf-8"))
    assert error["error_details"][0]["fatal"] is True
    assert error["error_details"][0]["retryable"] is False


def test_validation_error_retries_and_preserves_backend_response(tmp_path: Path) -> None:
    invalid_response = {"unexpected": "value"}
    backend = FakeLLMBackend([invalid_response, {"value": "recovered"}])
    client = StructuredLLMClient(backend, tmp_path, max_retries=2)

    response, trace = _call(client)

    assert response.value == "recovered"
    assert trace.attempts == 2
    assert len(backend.calls) == 2
    cache = json.loads(next(tmp_path.glob("tiny/*.json")).read_text(encoding="utf-8"))
    assert len(cache["previous_error_details"]) == 1
    detail = cache["previous_error_details"][0]
    assert detail["type"] == "ValidationError"
    assert detail["retryable"] is True
    assert detail["response_data"] == invalid_response
    assert detail["raw_response"] == {"fake": True, "data": invalid_response}
    assert detail["request_id"] == "fake-1"


def test_server_error_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "cir_qdiag_openai_cirr.models.client.time.sleep",
        lambda _: None,
    )
    backend = FakeLLMBackend([HTTPFailure(500, "temporary"), {"value": "ok"}])
    client = StructuredLLMClient(backend, tmp_path, max_retries=2)

    response, trace = _call(client)

    assert response.value == "ok"
    assert trace.attempts == 2
    cache = json.loads(next(tmp_path.glob("tiny/*.json")).read_text(encoding="utf-8"))
    assert cache["previous_error_details"] == [
        {
            "attempt": 1,
            "type": "HTTPFailure",
            "message": "temporary",
            "retryable": True,
            "fatal": False,
        }
    ]


def test_server_retry_after_is_respected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(
        "cir_qdiag_openai_cirr.models.client.time.sleep",
        lambda seconds: sleeps.append(seconds),
    )
    backend = FakeLLMBackend(
        [HTTPFailure(502, "origin bad gateway", retry_after=60), {"value": "ok"}]
    )
    response, trace = _call(
        StructuredLLMClient(backend, tmp_path, max_retries=2)
    )
    assert response.value == "ok"
    assert trace.attempts == 2
    assert sleeps == [60.0]


def test_backend_identity_is_part_of_cache_key(tmp_path: Path) -> None:
    first_backend = FakeLLMBackend(
        [{"value": "first"}], identity={"backend_contract": "fake", "max_pixels": 1}
    )
    first, first_trace = _call(StructuredLLMClient(first_backend, tmp_path, max_retries=0))
    second_backend = FakeLLMBackend(
        [{"value": "second"}], identity={"backend_contract": "fake", "max_pixels": 2}
    )
    second, second_trace = _call(StructuredLLMClient(second_backend, tmp_path, max_retries=0))

    assert first.value == "first"
    assert second.value == "second"
    assert first_trace.input_hash != second_trace.input_hash
    assert len(second_backend.calls) == 1


def test_semantically_rejected_cache_is_quarantined_and_regenerated(tmp_path: Path) -> None:
    backend = FakeLLMBackend([{"value": "bad-domain-value"}, {"value": "corrected"}])
    client = StructuredLLMClient(backend, tmp_path, max_retries=0)

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
