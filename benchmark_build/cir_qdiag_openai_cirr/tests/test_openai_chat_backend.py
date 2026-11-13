from __future__ import annotations

import json

import httpx
import pytest
from pydantic import BaseModel

from cir_qdiag_openai_cirr.models.base import BackendResponseError
from cir_qdiag_openai_cirr.models.openai_chat import (
    ChatCompletionsHTTPError,
    OpenAIChatBackend,
    _resolve_setting,
)


class TinyResponse(BaseModel):
    value: str


class FakeHTTPClient:
    def __init__(
        self,
        content: str = '{"value":"ok"}',
        refusal: str | None = None,
        *,
        string_usage: bool = False,
        string_response: bool = False,
        html_response: bool = False,
        status_code: int = 200,
        error_body: object | None = None,
        response_headers: dict[str, str] | None = None,
    ) -> None:
        self.content = content
        self.refusal = refusal
        self.string_usage = string_usage
        self.string_response = string_response
        self.html_response = html_response
        self.status_code = status_code
        self.error_body = error_body
        self.response_headers = response_headers or {}
        self.url: str | None = None
        self.payload: dict | None = None

    def post(self, url: str, *, json: dict) -> httpx.Response:
        self.url = url
        self.payload = json
        request = httpx.Request("POST", url)
        if self.html_response:
            return httpx.Response(
                self.status_code,
                text="<!doctype html><html><body>gateway UI</body></html>",
                headers=self.response_headers,
                request=request,
            )
        if self.status_code >= 400:
            return httpx.Response(
                self.status_code,
                json=self.error_body,
                headers=self.response_headers,
                request=request,
            )
        raw: object = {
            "id": "resp_123",
            "choices": [
                {"message": {"content": self.content, "refusal": self.refusal}}
            ],
            "usage": (
                "usage temporarily unavailable"
                if self.string_usage
                else {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}
            ),
        }
        if self.string_response:
            raw = json_module_dumps(raw)
        return httpx.Response(
            self.status_code,
            json=raw,
            headers=self.response_headers,
            request=request,
        )


def json_module_dumps(value: object) -> str:
    return json.dumps(value)


def _backend(client: FakeHTTPClient) -> OpenAIChatBackend:
    backend = OpenAIChatBackend.__new__(OpenAIChatBackend)
    backend.api_base = "https://ai.gs88.shop/v1"
    backend.chat_completions_url = f"{backend.api_base}/chat/completions"
    backend.generation_model = "gpt-5.6-terra"
    backend.validation_model = "gpt-5.6-terra"
    backend.generation_reasoning_effort = "medium"
    backend.validation_reasoning_effort = "medium"
    backend.structured_output_mode = "json_schema"
    backend.image_detail = "high"
    backend.model_name = "stage-routed-openai-chat"
    backend.model_revision = None
    backend.client = client
    return backend


def test_constructor_uses_curl_compatible_minimal_httpx_client(monkeypatch) -> None:
    captured: dict = {}
    sentinel = object()

    def fake_client(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        "cir_qdiag_openai_cirr.models.openai_chat.httpx.Client",
        fake_client,
    )
    backend = OpenAIChatBackend(
        api_base="https://ai.gs88.shop/v1",
        api_key="secret",
        generation_model="gpt-5.6-terra",
        validation_model="gpt-5.6-terra",
        generation_reasoning_effort=None,
        validation_reasoning_effort=None,
    )

    assert backend.client is sentinel
    assert backend.chat_completions_url == (
        "https://ai.gs88.shop/v1/chat/completions"
    )
    assert captured["trust_env"] is False
    assert captured["headers"] == {
        "Authorization": "Bearer secret",
        "Content-Type": "application/json",
        "Accept": "*/*",
        "User-Agent": "curl/8.0",
    }


def test_text_chat_completion_uses_json_schema_and_reasoning() -> None:
    client = FakeHTTPClient()
    backend = _backend(client)

    result = backend.generate_json(
        [],
        "system",
        "user",
        TinyResponse,
        stage="validate_visual",
        max_output_tokens=100,
    )

    assert result.data == {"value": "ok"}
    assert result.model_name == "gpt-5.6-terra"
    assert result.reasoning_effort == "medium"
    assert result.request_id == "resp_123"
    assert result.usage["total_tokens"] == 14
    assert client.url == "https://ai.gs88.shop/v1/chat/completions"
    assert client.payload["reasoning_effort"] == "medium"
    assert client.payload["max_tokens"] == 100
    response_format = client.payload["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"] == TinyResponse.model_json_schema()
    assert client.payload["messages"][1]["content"] == "user"


def test_multimodal_request_sends_numbered_image_data_urls(tmp_path) -> None:
    reference = tmp_path / "reference.png"
    target = tmp_path / "target.jpg"
    reference.write_bytes(b"reference")
    target.write_bytes(b"target")
    client = FakeHTTPClient()
    backend = _backend(client)

    backend.generate_json(
        [str(reference), str(target)],
        "system",
        "user",
        TinyResponse,
        stage="validate_visual",
    )

    content = client.payload["messages"][1]["content"]
    assert content[0] == {"type": "text", "text": "user"}
    assert content[1] == {"type": "text", "text": "Image 1:"}
    assert content[2]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[2]["image_url"]["detail"] == "high"
    assert content[3] == {"type": "text", "text": "Image 2:"}
    assert content[4]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_stage_routing_uses_configured_models() -> None:
    backend = _backend(FakeHTTPClient())
    assert backend._route("generate_add") == ("gpt-5.6-terra", "medium")
    assert backend._route("validate_visual") == ("gpt-5.6-terra", "medium")
    assert backend._route("generate_positive_paraphrases") == (
        "gpt-5.6-terra",
        "medium",
    )
    assert backend._route("review_positive_paraphrases") == (
        "gpt-5.6-terra",
        "medium",
    )


def test_text_only_request_uses_plain_string_content() -> None:
    client = FakeHTTPClient()
    backend = _backend(client)
    backend.generate_json(
        [],
        "system",
        "plain text request",
        TinyResponse,
        stage="generate_add",
    )
    assert client.payload["messages"][1]["content"] == "plain text request"


def test_prompt_structured_mode_omits_provider_specific_parameters() -> None:
    client = FakeHTTPClient()
    backend = _backend(client)
    backend.structured_output_mode = "prompt"
    backend.generation_reasoning_effort = None

    result = backend.generate_json(
        [],
        "system",
        "plain text request",
        TinyResponse,
        stage="generate_add",
    )

    assert result.data == {"value": "ok"}
    assert result.reasoning_effort is None
    assert "response_format" not in client.payload
    assert "reasoning_effort" not in client.payload
    system_prompt = client.payload["messages"][0]["content"]
    assert "JSON Schema" in system_prompt
    assert '"value"' in system_prompt


def test_string_usage_from_provider_does_not_discard_content() -> None:
    backend = _backend(FakeHTTPClient(string_usage=True))
    result = backend.generate_json(
        [],
        "system",
        "user",
        TinyResponse,
        stage="generate_add",
    )
    assert result.raw_response["usage"] == "usage temporarily unavailable"
    assert result.usage == {
        "raw": "usage temporarily unavailable",
        "compatibility_warning": "provider returned a non-object usage field",
    }


def test_json_string_completion_envelope_is_unwrapped() -> None:
    backend = _backend(FakeHTTPClient(string_response=True))
    result = backend.generate_json(
        [],
        "system",
        "user",
        TinyResponse,
        stage="generate_add",
    )
    assert result.data == {"value": "ok"}
    assert result.request_id == "resp_123"
    assert result.usage["total_tokens"] == 14


def test_html_200_response_reports_wrong_api_base() -> None:
    backend = _backend(FakeHTTPClient(html_response=True))
    with pytest.raises(BackendResponseError, match="missing its /v1 path") as caught:
        backend.generate_json(
            [],
            "system",
            "user",
            TinyResponse,
            stage="generate_add",
        )
    assert caught.value.retryable is False


def test_http_error_preserves_status_body_and_headers() -> None:
    backend = _backend(
        FakeHTTPClient(
            status_code=502,
            error_body={"error": {"retry_after": 60}},
            response_headers={"retry-after": "60"},
        )
    )
    with pytest.raises(ChatCompletionsHTTPError) as caught:
        backend.generate_json([], "system", "user", TinyResponse, stage="generate_add")
    assert caught.value.status_code == 502
    assert caught.value.body == {"error": {"retry_after": 60}}
    assert caught.value.response.headers["retry-after"] == "60"


def test_cloudflare_html_error_is_concise_and_uses_safe_retry_delay() -> None:
    backend = _backend(FakeHTTPClient(html_response=True, status_code=502))
    with pytest.raises(ChatCompletionsHTTPError) as caught:
        backend.generate_json([], "system", "user", TinyResponse, stage="generate_add")
    assert str(caught.value) == "Error code: 502 - HTML error response"
    assert caught.value.retry_after == 60
    assert "<!doctype html>" in caught.value.body


def test_refusal_preserves_raw_response() -> None:
    backend = _backend(FakeHTTPClient(content="", refusal="Cannot comply."))
    with pytest.raises(BackendResponseError, match="Cannot comply") as caught:
        backend.generate_json([], "system", "user", TinyResponse, stage="validate_visual")
    assert caught.value.request_id == "resp_123"
    assert caught.value.raw_response["id"] == "resp_123"


def test_invalid_json_preserves_complete_response() -> None:
    malformed = '{"value":'
    backend = _backend(FakeHTTPClient(content=malformed))
    with pytest.raises(BackendResponseError) as caught:
        backend.generate_json([], "system", "user", TinyResponse, stage="validate_visual")
    assert caught.value.response_text == malformed
    assert json.dumps(caught.value.raw_response)


def test_environment_placeholder_must_be_filled(monkeypatch) -> None:
    monkeypatch.delenv("MISSING_OPENAI_VALUE", raising=False)
    with pytest.raises(ValueError, match="export MISSING_OPENAI_VALUE"):
        _resolve_setting("${MISSING_OPENAI_VALUE}", "test setting")
    monkeypatch.setenv("MISSING_OPENAI_VALUE", "filled")
    assert _resolve_setting("${MISSING_OPENAI_VALUE}", "test setting") == "filled"
