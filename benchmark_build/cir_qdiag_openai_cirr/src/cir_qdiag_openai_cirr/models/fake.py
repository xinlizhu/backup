from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import Any, Iterable

from pydantic import BaseModel

from cir_qdiag_openai_cirr.schemas import BackendResult


class FakeLLMBackend:
    def __init__(
        self,
        responses: Iterable[dict[str, Any] | Exception],
        *,
        identity: dict[str, Any] | None = None,
    ):
        self.responses = deque(
            item if isinstance(item, Exception) else deepcopy(item) for item in responses
        )
        self.calls: list[dict[str, Any]] = []
        self.identity = identity or {"backend_contract": "fake-v1"}

    def cache_identity(self) -> dict[str, Any]:
        return dict(self.identity)

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
        self.calls.append(
            {
                "images": images,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "json_schema": response_model.model_json_schema(),
                "response_model": response_model,
                "stage": stage,
                "max_output_tokens": max_output_tokens,
            }
        )
        if not self.responses:
            raise RuntimeError(f"FakeLLMBackend has no response left for stage {stage}")
        data = self.responses.popleft()
        if isinstance(data, Exception):
            raise data
        return BackendResult(
            data=data,
            raw_response={"fake": True, "data": data},
            model_name="fake-llm",
            model_revision="test",
            request_id=f"fake-{len(self.calls)}",
            provider="fake",
        )
