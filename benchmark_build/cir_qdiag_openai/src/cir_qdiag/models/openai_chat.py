from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel

from cir_qdiag.models.base import BackendResponseError
from cir_qdiag.schemas import BackendResult


_ENV_REFERENCE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


class ChatCompletionsHTTPError(RuntimeError):
    """HTTP failure carrying the fields used by the pipeline retry policy."""

    def __init__(self, response: httpx.Response, body: Any) -> None:
        self.response = response
        self.status_code = response.status_code
        self.body = body
        self.retry_after = (
            60
            if response.status_code in {502, 503, 504}
            and isinstance(body, str)
            and _looks_like_html(body)
            else None
        )
        super().__init__(
            f"Error code: {response.status_code} - {_response_body_summary(body)}"
        )


class OpenAIChatBackend:
    """Stage-routed Chat Completions backend with locally validated JSON output."""

    def __init__(
        self,
        *,
        api_base: str,
        api_key: str,
        generation_model: str,
        selection_model: str,
        generation_reasoning_effort: str | None,
        selection_reasoning_effort: str | None,
        structured_output_mode: str = "prompt",
        image_detail: str = "high",
        timeout_seconds: float = 300.0,
        organization: str | None = None,
        project: str | None = None,
    ) -> None:
        self.api_base = _resolve_setting(api_base, "OpenAI API URL").rstrip("/")
        resolved_key = _resolve_setting(api_key, "OpenAI API key")
        self.generation_model = generation_model
        self.selection_model = selection_model
        self.generation_reasoning_effort = generation_reasoning_effort
        self.selection_reasoning_effort = selection_reasoning_effort
        self.structured_output_mode = structured_output_mode
        self.image_detail = image_detail
        self.model_name = "stage-routed-openai-chat"
        self.model_revision = None

        headers = {
            "Authorization": f"Bearer {resolved_key}",
            "Content-Type": "application/json",
            "Accept": "*/*",
            # This provider returns 502 for the OpenAI SDK's Stainless headers.
            # Match the minimal curl request that is known to work.
            "User-Agent": "curl/8.0",
        }
        if organization:
            headers["OpenAI-Organization"] = organization
        if project:
            headers["OpenAI-Project"] = project
        self.chat_completions_url = f"{self.api_base}/chat/completions"
        self.client = httpx.Client(
            headers=headers,
            timeout=httpx.Timeout(timeout_seconds),
            trust_env=False,
        )

    def cache_identity(self) -> dict[str, Any]:
        return {
            "backend_contract": "minimal-httpx-chat-structured-v3",
            "api_base": self.api_base,
            "structured_output_mode": self.structured_output_mode,
            "generation_model": self.generation_model,
            "selection_model": self.selection_model,
            "generation_reasoning_effort": self.generation_reasoning_effort,
            "selection_reasoning_effort": self.selection_reasoning_effort,
            "image_detail": self.image_detail,
        }

    def generate_json(
        self,
        images: list[str],
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
        *,
        stage: str,
        max_output_tokens: int = 1024,
    ) -> BackendResult:
        model_name, effort = self._route(stage)
        if images:
            content: str | list[dict[str, Any]] = [
                {"type": "text", "text": user_prompt}
            ]
            for index, image_path in enumerate(images, start=1):
                content.extend(
                    [
                        {"type": "text", "text": f"Image {index}:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": _image_data_url(image_path),
                                "detail": self.image_detail,
                            },
                        },
                    ]
                )
        else:
            # Some OpenAI-compatible gateways fail to route a text-only
            # request when its content is encoded as multimodal content parts.
            content = user_prompt

        schema = response_model.model_json_schema()
        if self.structured_output_mode == "prompt":
            system_prompt = (
                f"{system_prompt}\n\n"
                "Return only one JSON object that validates against this JSON Schema. "
                "Do not use Markdown fences or add commentary.\n"
                f"{json.dumps(schema, ensure_ascii=False, separators=(',', ':'))}"
            )
        elif self.structured_output_mode != "json_schema":
            raise ValueError(
                f"unsupported structured output mode: {self.structured_output_mode}"
            )

        request: dict[str, Any] = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "max_tokens": max_output_tokens,
        }
        if self.structured_output_mode == "json_schema":
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": _schema_name(stage),
                    "strict": True,
                    "schema": schema,
                },
            }
        if effort is not None:
            request["reasoning_effort"] = effort

        response = self.client.post(self.chat_completions_url, json=request)
        response_body = _decode_http_response(response)
        if response.status_code >= 400:
            raise ChatCompletionsHTTPError(response, response_body)

        response_view = _decode_response_container(response_body)
        raw_value = _safe_json_value(response_view)
        raw_response = (
            raw_value
            if isinstance(raw_value, (dict, str))
            else {"raw_response": raw_value}
        )
        request_id = _field(response_view, "id") or response.headers.get("x-request-id")
        if isinstance(raw_response, str) and _looks_like_html(raw_response):
            raise BackendResponseError(
                "Chat Completions endpoint returned an HTML website; "
                "the API base URL is likely missing its /v1 path",
                response_text="",
                raw_response=raw_response,
                request_id=request_id if isinstance(request_id, str) else None,
                model_name=model_name,
                model_revision=None,
                retryable=False,
            )
        choices = _field(response_view, "choices")
        if not isinstance(choices, (list, tuple)) or not choices:
            raise BackendResponseError(
                "Chat Completions response has no choices",
                response_text="",
                raw_response=raw_response,
                request_id=request_id if isinstance(request_id, str) else None,
                model_name=model_name,
                model_revision=None,
            )
        message = _field(choices[0], "message")
        response_text_value = _field(message, "content")
        response_text = (
            response_text_value if isinstance(response_text_value, str) else ""
        )
        refusal = _field(message, "refusal")
        if refusal:
            raise BackendResponseError(
                f"Chat Completions refusal: {refusal}",
                response_text=response_text,
                raw_response=raw_response,
                request_id=request_id if isinstance(request_id, str) else None,
                model_name=model_name,
                model_revision=None,
            )
        try:
            data = _extract_json(response_text)
            validated = response_model.model_validate(data)
        except (ValueError, json.JSONDecodeError) as exc:
            raise BackendResponseError(
                f"Chat Completions output failed {response_model.__name__} validation: {exc}",
                response_text=response_text,
                raw_response=raw_response,
                request_id=request_id if isinstance(request_id, str) else None,
                model_name=model_name,
                model_revision=None,
            ) from exc
        usage_value = _safe_json_value(_field(response_view, "usage"))
        usage = (
            usage_value
            if isinstance(usage_value, dict)
            else (
                {
                    "raw": usage_value,
                    "compatibility_warning": (
                        "provider returned a non-object usage field"
                    ),
                }
                if usage_value is not None
                else None
            )
        )
        return BackendResult(
            data=validated.model_dump(mode="json"),
            raw_response=raw_response,
            model_name=model_name,
            request_id=request_id if isinstance(request_id, str) else None,
            provider="openai-compatible-chat",
            reasoning_effort=effort,
            usage=usage,
        )

    def _route(self, stage: str) -> tuple[str, str | None]:
        if stage == "validate_group":
            return self.selection_model, self.selection_reasoning_effort
        return self.generation_model, self.generation_reasoning_effort


def _resolve_setting(value: str, label: str) -> str:
    stripped = value.strip()
    match = _ENV_REFERENCE.fullmatch(stripped)
    if match:
        env_name = match.group(1)
        stripped = os.environ.get(env_name, "").strip()
        if not stripped:
            raise ValueError(f"{label} is unset; export {env_name} before running")
    if not stripped or stripped.startswith("YOUR_"):
        raise ValueError(f"{label} still contains a placeholder")
    return stripped


def _image_data_url(image_path: str) -> str:
    absolute = Path(image_path).expanduser().resolve()
    if not absolute.is_file():
        raise FileNotFoundError(f"OpenAI image does not exist: {absolute}")
    mime_type = mimetypes.guess_type(absolute.name)[0]
    if mime_type not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
        raise ValueError(f"unsupported OpenAI image format for {absolute}")
    encoded = base64.b64encode(absolute.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _schema_name(stage: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_-]", "_", stage)
    return (value or "cir_qdiag_response")[:64]


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("model response does not contain a JSON object")
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("model JSON response must be an object")
    return value


def _field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _decode_http_response(response: httpx.Response) -> Any:
    try:
        return response.json()
    except (json.JSONDecodeError, ValueError):
        return response.text


def _decode_response_container(value: Any) -> Any:
    """Unwrap providers that return the Chat Completions envelope as a JSON string."""

    current = value
    for _ in range(2):
        if not isinstance(current, str):
            break
        try:
            decoded = json.loads(current)
        except json.JSONDecodeError:
            break
        if decoded == current:
            break
        current = decoded
    return current


def _looks_like_html(value: str) -> bool:
    stripped = value.lstrip().casefold()
    return stripped.startswith("<!doctype html") or stripped.startswith("<html")


def _response_body_summary(value: Any) -> str:
    if isinstance(value, str) and _looks_like_html(value):
        title = re.search(r"<title>(.*?)</title>", value, flags=re.IGNORECASE | re.DOTALL)
        if title:
            normalized = " ".join(title.group(1).split())
            return f"HTML response: {normalized}"
        return "HTML error response"
    rendered = repr(value)
    return rendered if len(rendered) <= 1000 else f"{rendered[:1000]}..."


def _safe_json_value(value: Any, _seen: set[int] | None = None) -> Any:
    """Serialize SDK objects even when an OpenAI-compatible field has the wrong type."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    seen = _seen if _seen is not None else set()
    identity = id(value)
    if identity in seen:
        return "<recursive>"
    seen.add(identity)
    if isinstance(value, dict):
        return {
            str(key): _safe_json_value(item, seen)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_safe_json_value(item, seen) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _safe_json_value(model_dump(mode="json"), seen)
        except Exception:
            pass
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return {
            str(key): _safe_json_value(item, seen)
            for key, item in attributes.items()
            if not str(key).startswith("_")
        }
    return repr(value)
