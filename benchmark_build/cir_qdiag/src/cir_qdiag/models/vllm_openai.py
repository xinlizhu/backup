from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cir_qdiag.models.base import BackendResponseError
from cir_qdiag.schemas import BackendResult
from cir_qdiag.utils.hashing import local_model_fingerprint, sha256_text, stable_json


class VLLMOpenAIBackend:
    def __init__(
        self,
        model_name: str,
        model_source: str,
        api_base: str,
        api_key: str = "EMPTY",
        timeout_seconds: float = 300.0,
        model_revision: str | None = None,
        disable_thinking: bool = True,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("install the 'openai' package to use the vLLM backend") from exc
        self.model_name = model_name
        self.model_source = model_source
        self.model_revision = model_revision
        self.api_base = api_base
        self.disable_thinking = disable_thinking
        self.mm_processor_kwargs = dict(mm_processor_kwargs or {})
        self.model_artifact_fingerprint = local_model_fingerprint(model_source)
        self.top_k = 20
        self.client = OpenAI(
            base_url=api_base,
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=0,
        )

    def cache_identity(self) -> dict[str, Any]:
        return {
            "backend_contract": "vllm-openai-v4",
            "model_name": self.model_name,
            "model_source": self.model_source,
            "model_revision": self.model_revision,
            "model_artifact_fingerprint": getattr(self, "model_artifact_fingerprint", None),
            "api_base": self.api_base,
            "disable_thinking": self.disable_thinking,
            "mm_processor_kwargs": self.mm_processor_kwargs,
            "top_k": self.top_k,
        }

    def generate_json(
        self,
        images: list[str],
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any],
        *,
        stage: str,
        temperature: float = 0.0,
        seed: int | None = None,
        max_new_tokens: int = 2048,
    ) -> BackendResult:
        content: list[dict[str, Any]] = []
        for index, image_path in enumerate(images, start=1):
            absolute = Path(image_path).resolve()
            if not absolute.is_file():
                raise FileNotFoundError(f"VLM image does not exist: {absolute}")
            stat = absolute.stat()
            media_identity = sha256_text(
                stable_json(
                    {
                        "path": str(absolute),
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                        "mm_processor_kwargs": self.mm_processor_kwargs,
                    }
                )
            )
            content.extend(
                [
                    {"type": "text", "text": f"Image {index}:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"file://{absolute}"},
                        "uuid": f"cir-qdiag:{media_identity}",
                    },
                ]
            )
        content.append({"type": "text", "text": user_prompt})
        extra_body: dict[str, Any] = {"top_k": self.top_k}
        if self.disable_thinking:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        if images and self.mm_processor_kwargs:
            extra_body["mm_processor_kwargs"] = self.mm_processor_kwargs
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            max_completion_tokens=max_new_tokens,
            temperature=temperature,
            top_p=1.0 if temperature == 0 else 0.8,
            seed=seed,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": stage.replace("-", "_"),
                    "strict": True,
                    "schema": json_schema,
                },
            },
            extra_body=extra_body,
        )
        text = response.choices[0].message.content or ""
        raw_response = response.model_dump(mode="json")
        try:
            data = _extract_json(text)
        except ValueError as exc:
            raise BackendResponseError(
                str(exc),
                response_text=text,
                raw_response=raw_response,
                request_id=getattr(response, "id", None),
                model_name=self.model_name,
                model_revision=self.model_revision,
            ) from exc
        return BackendResult(
            data=data,
            raw_response=raw_response,
            model_name=self.model_name,
            model_revision=self.model_revision,
            request_id=getattr(response, "id", None),
        )


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"model did not return a JSON object: {stripped[:500]}")
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("model JSON response must be an object")
    return value
