from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import Any, Iterable

from cir_qdiag.schemas import BackendResult


class FakeVLMBackend:
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
        json_schema: dict[str, Any],
        *,
        stage: str,
        temperature: float = 0.0,
        seed: int | None = None,
        max_new_tokens: int = 1024,
    ) -> BackendResult:
        self.calls.append(
            {
                "images": images,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "json_schema": json_schema,
                "stage": stage,
                "temperature": temperature,
                "seed": seed,
            }
        )
        if not self.responses:
            raise RuntimeError(f"FakeVLMBackend has no response left for stage {stage}")
        data = self.responses.popleft()
        if isinstance(data, Exception):
            raise data
        return BackendResult(
            data=data,
            raw_response={"fake": True, "data": data},
            model_name="fake-vlm",
            model_revision="test",
            request_id=f"fake-{len(self.calls)}",
        )
