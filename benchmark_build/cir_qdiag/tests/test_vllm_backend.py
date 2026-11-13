from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from cir_qdiag.models.client import StructuredVLMClient
from cir_qdiag.models.vllm_openai import VLLMOpenAIBackend


class FakeCompletions:
    def __init__(self, content: str = '{"value":"ok"}') -> None:
        self.kwargs = None
        self.content = content

    def create(self, **kwargs):
        self.kwargs = kwargs
        raw_response = {
            "id": "request-1",
            "choices": [{"message": {"content": self.content}}],
        }
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
            id="request-1",
            model_dump=lambda **_: raw_response,
        )


class TinyResponse(BaseModel):
    value: str


def test_multimodal_processor_limit_is_sent_per_request(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    completions = FakeCompletions()
    backend = VLLMOpenAIBackend.__new__(VLLMOpenAIBackend)
    backend.model_name = "model"
    backend.model_source = "/model"
    backend.model_revision = None
    backend.api_base = "http://localhost/v1"
    backend.disable_thinking = True
    backend.mm_processor_kwargs = {"max_pixels": 4_194_304}
    backend.top_k = 20
    backend.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    result = backend.generate_json(
        [str(image)],
        "system",
        "user",
        {"type": "object"},
        stage="test",
        temperature=0.0,
        seed=42,
        max_new_tokens=16,
    )

    assert result.data == {"value": "ok"}
    assert completions.kwargs["extra_body"]["mm_processor_kwargs"] == {"max_pixels": 4_194_304}
    assert completions.kwargs["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
    content = completions.kwargs["messages"][1]["content"]
    image_item = next(item for item in content if item["type"] == "image_url")
    assert image_item["image_url"]["url"] == f"file://{image.resolve()}"
    assert image_item["uuid"].startswith("cir-qdiag:")


def test_unparseable_json_preserves_complete_raw_response(tmp_path: Path) -> None:
    malformed = '{"value":"' + ("x" * 700) + '"'
    completions = FakeCompletions(malformed)
    backend = VLLMOpenAIBackend.__new__(VLLMOpenAIBackend)
    backend.model_name = "model"
    backend.model_source = "/model"
    backend.model_revision = "revision"
    backend.api_base = "http://localhost/v1"
    backend.disable_thinking = True
    backend.mm_processor_kwargs = {}
    backend.top_k = 20
    backend.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client = StructuredVLMClient(backend, tmp_path, max_retries=0)

    with pytest.raises(RuntimeError, match="BackendResponseError"):
        client.call(
            TinyResponse,
            images=[],
            system_prompt="system",
            user_prompt='{"input":"x"}',
            stage="malformed",
            temperature=0.0,
            seed=42,
            max_new_tokens=16,
        )

    error_path = next(tmp_path.glob("malformed/*.error.json"))
    error = json.loads(error_path.read_text(encoding="utf-8"))
    detail = error["error_details"][0]
    assert detail["response_text"] == malformed
    assert detail["raw_response"] == {
        "id": "request-1",
        "choices": [{"message": {"content": malformed}}],
    }
    assert detail["request_id"] == "request-1"
    assert detail["model_name"] == "model"
    assert detail["model_revision"] == "revision"
    assert detail["effective_seed"] == 42
