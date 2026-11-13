from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from cir_qdiag_openai_cirr.models.base import (
    BackendResponseError,
    FatalProviderError,
    StructuredLanguageBackend,
)
from cir_qdiag_openai_cirr.schemas import CallTrace
from cir_qdiag_openai_cirr.utils.hashing import input_fingerprint, sha256_text

LOGGER = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class StructuredLLMClient:
    def __init__(
        self,
        backend: StructuredLanguageBackend,
        cache_dir: str | Path,
        prompt_version: str = "v8",
        max_retries: int = 2,
    ) -> None:
        self.backend = backend
        self.cache_dir = Path(cache_dir)
        self.prompt_version = prompt_version
        self.max_retries = max_retries

    def call(
        self,
        response_model: type[T],
        *,
        images: list[str],
        system_prompt: str,
        user_prompt: str,
        stage: str,
        max_output_tokens: int,
    ) -> tuple[T, CallTrace]:
        schema = response_model.model_json_schema()
        prompt_hash = sha256_text(system_prompt + "\n---\n" + user_prompt)
        request_payload = {
            "stage": stage,
            "backend": type(self.backend).__name__,
            "model_name": getattr(self.backend, "model_name", None),
            "model_revision": getattr(self.backend, "model_revision", None),
            "backend_identity": _backend_identity(self.backend),
            "prompt_hash": prompt_hash,
            "prompt_version": self.prompt_version,
            "schema": schema,
            "max_output_tokens": max_output_tokens,
        }
        input_hash = input_fingerprint(request_payload, images)
        cache_path = self.cache_dir / stage / f"{input_hash}.json"
        error_path = cache_path.with_suffix(".error.json")
        if cache_path.is_file():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            parsed = response_model.model_validate(cached["parsed"])
            trace = CallTrace.model_validate(cached["trace"])
            trace.cached = True
            return parsed, trace

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        errors: list[str] = []
        error_details: list[dict[str, object]] = []
        for attempt in range(self.max_retries + 1):
            result = None
            try:
                result = self.backend.generate_json(
                    images=images,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response_model=response_model,
                    stage=stage,
                    max_output_tokens=max_output_tokens,
                )
                parsed = response_model.model_validate(result.data)
                trace = CallTrace(
                    stage=stage,
                    model_name=result.model_name,
                    model_revision=result.model_revision,
                    provider=result.provider,
                    reasoning_effort=result.reasoning_effort,
                    prompt_version=self.prompt_version,
                    prompt_hash=prompt_hash,
                    input_hash=input_hash,
                    raw_response_path=str(cache_path),
                    attempts=attempt + 1,
                    request_id=result.request_id,
                    usage=result.usage,
                )
                payload = {
                    "request": request_payload,
                    "images": images,
                    "parsed": parsed.model_dump(mode="json"),
                    "raw_response": result.raw_response,
                    "request_id": result.request_id,
                    "trace": trace.model_dump(mode="json"),
                    "attempt": attempt + 1,
                    "previous_errors": errors,
                    "previous_error_details": error_details,
                }
                _write_json_atomic(cache_path, payload)
                error_path.unlink(missing_ok=True)
                return parsed, trace
            except Exception as exc:
                message = f"attempt {attempt + 1}: {type(exc).__name__}: {exc}"
                errors.append(message)
                retryable = _is_retryable_error(exc)
                fatal = _is_fatal_provider_error(exc)
                detail: dict[str, object] = {
                    "attempt": attempt + 1,
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "retryable": retryable,
                    "fatal": fatal,
                }
                if result is not None:
                    detail.update(
                        {
                            "response_data": result.data,
                            "raw_response": result.raw_response,
                            "request_id": result.request_id,
                            "model_name": result.model_name,
                            "model_revision": result.model_revision,
                        }
                    )
                elif isinstance(exc, BackendResponseError):
                    detail.update(
                        {
                            "response_text": exc.response_text,
                            "raw_response": exc.raw_response,
                            "request_id": exc.request_id,
                            "model_name": exc.model_name,
                            "model_revision": exc.model_revision,
                        }
                    )
                error_details.append(detail)
                _write_json_atomic(
                    error_path,
                    {
                        "request": request_payload,
                        "images": images,
                        "errors": errors,
                        "error_details": error_details,
                    },
                )
                LOGGER.warning("LLM %s failed: %s", stage, message)
                if fatal:
                    raise FatalProviderError(
                        f"LLM stage {stage} cannot continue because the provider "
                        f"account or quota is unavailable: {type(exc).__name__}: {exc}"
                    ) from exc
                if not retryable or attempt >= self.max_retries:
                    break
                delay = _retry_delay_seconds(exc, attempt)
                LOGGER.warning(
                    "retrying LLM %s in %.1f seconds",
                    stage,
                    delay,
                )
                time.sleep(delay)
        _write_json_atomic(
            error_path,
            {
                "request": request_payload,
                "images": images,
                "errors": errors,
                "error_details": error_details,
            },
        )
        raise RuntimeError(
            f"LLM stage {stage} failed after {len(errors)} attempt(s): {'; '.join(errors)}"
        )

    def reject_cached_response(self, trace: CallTrace, reason: str) -> Path | None:
        """Quarantine a schema-valid response that failed caller-level semantic checks."""

        cache_path = Path(trace.raw_response_path)
        if not cache_path.is_file():
            return None
        cache_root = self.cache_dir.resolve()
        resolved = cache_path.resolve()
        if not resolved.is_relative_to(cache_root):
            raise ValueError(f"refusing to reject cache outside {cache_root}: {resolved}")
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        payload["semantic_rejection"] = {"reason": reason}
        fingerprint = sha256_text(
            json.dumps(payload.get("raw_response"), ensure_ascii=False, sort_keys=True)
        )[:12]
        rejected_path = cache_path.with_name(f"{cache_path.stem}.rejected.{fingerprint}.json")
        _write_json_atomic(rejected_path, payload)
        cache_path.unlink(missing_ok=True)
        return rejected_path


def _write_json_atomic(path: Path, value: object) -> None:
    temp = path.with_suffix(path.suffix + f".tmp.{threading.get_ident()}")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _backend_identity(backend: StructuredLanguageBackend) -> dict[str, object]:
    identity = getattr(backend, "cache_identity", None)
    if callable(identity):
        value = identity()
        if not isinstance(value, dict):
            raise TypeError("backend cache_identity() must return a dict")
        return value
    return {"backend_contract": type(backend).__name__}


def _is_retryable_error(exc: Exception) -> bool:
    explicit = getattr(exc, "retryable", None)
    if isinstance(explicit, bool):
        return explicit
    if isinstance(exc, ValidationError):
        # Inside call(), ValidationError means the backend produced JSON that
        # did not satisfy the requested response schema. A new deterministic
        # retry seed gives the model a chance to repair that protocol error.
        return True
    if isinstance(exc, (FileNotFoundError, PermissionError)):
        return False
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        if status_code in {408, 409, 429}:
            return True
        if 400 <= status_code < 500:
            return False
        return status_code >= 500
    return True


def _is_fatal_provider_error(exc: Exception) -> bool:
    """Identify account-wide failures that cannot improve on another sample."""

    if isinstance(exc, FatalProviderError):
        return True
    status_code = getattr(exc, "status_code", None)
    if status_code in {401, 402}:
        return True

    body = getattr(exc, "body", None)
    searchable = _searchable_error_text(body, exc)
    fatal_markers = (
        "daily_limit_exceeded",
        "insufficient_quota",
        "quota_exceeded",
        "billing_hard_limit",
        "credit_balance",
        "credits_exhausted",
        "subscription_limit",
        "subscription quota",
        "account balance",
        "额度不足",
        "额度已用完",
        "订阅额度",
        "余额不足",
        "手动重置额度",
    )
    return any(marker in searchable for marker in fatal_markers)


def _searchable_error_text(body: object, exc: Exception) -> str:
    try:
        body_text = json.dumps(body, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        body_text = str(body)
    return f"{type(exc).__name__} {exc} {body_text}".casefold()


def _retry_delay_seconds(exc: Exception, attempt: int) -> float:
    parsed = _positive_seconds(getattr(exc, "retry_after", None))
    if parsed is not None:
        return min(parsed, 300.0)
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        try:
            value = headers.get("retry-after")
        except Exception:
            value = None
        parsed = _positive_seconds(value)
        if parsed is not None:
            return min(parsed, 300.0)
    parsed = _find_retry_after(getattr(exc, "body", None))
    if parsed is not None:
        return min(parsed, 300.0)
    return float(min(2**attempt, 30))


def _find_retry_after(value: object) -> float | None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).casefold().replace("-", "_")
            if normalized == "retry_after":
                parsed = _positive_seconds(item)
                if parsed is not None:
                    return parsed
            nested = _find_retry_after(item)
            if nested is not None:
                return nested
    elif isinstance(value, (list, tuple)):
        for item in value:
            nested = _find_retry_after(item)
            if nested is not None:
                return nested
    return None


def _positive_seconds(value: object) -> float | None:
    try:
        seconds = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return seconds if seconds > 0 else None
