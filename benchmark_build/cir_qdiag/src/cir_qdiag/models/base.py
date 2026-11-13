from __future__ import annotations

from typing import Any, Protocol

from cir_qdiag.schemas import BackendResult


class BackendResponseError(ValueError):
    """A model response was received but could not be converted to structured data."""

    def __init__(
        self,
        message: str,
        *,
        response_text: str,
        raw_response: dict[str, Any] | str,
        request_id: str | None,
        model_name: str,
        model_revision: str | None,
    ) -> None:
        super().__init__(message)
        self.response_text = response_text
        self.raw_response = raw_response
        self.request_id = request_id
        self.model_name = model_name
        self.model_revision = model_revision


class VisionLanguageBackend(Protocol):
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
    ) -> BackendResult: ...
