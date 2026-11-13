from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from cir_qdiag.models.base import BackendResponseError, VisionLanguageBackend
from cir_qdiag.schemas import CallTrace
from cir_qdiag.utils.hashing import input_fingerprint, sha256_text

LOGGER = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class StructuredVLMClient:
    def __init__(
        self,
        backend: VisionLanguageBackend,
        cache_dir: str | Path,
        prompt_version: str = "v5",
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
        temperature: float,
        seed: int | None,
        max_new_tokens: int,
    ) -> tuple[T, CallTrace]:
        schema = response_model.model_json_schema()
        prompt_hash = sha256_text(system_prompt + "\n---\n" + user_prompt)
        request_payload = {
            "stage": stage,
            "backend": type(self.backend).__name__,
            "model_name": getattr(
                self.backend, "model_name", getattr(self.backend, "name_or_path", None)
            ),
            "model_revision": getattr(self.backend, "model_revision", None),
            "backend_identity": _backend_identity(self.backend),
            "prompt_hash": prompt_hash,
            "prompt_version": self.prompt_version,
            "schema": schema,
            "temperature": temperature,
            "seed": seed,
            "max_new_tokens": max_new_tokens,
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
            effective_seed = None if seed is None else seed + attempt
            result = None
            try:
                result = self.backend.generate_json(
                    images=images,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    json_schema=schema,
                    stage=stage,
                    temperature=temperature,
                    seed=effective_seed,
                    max_new_tokens=max_new_tokens,
                )
                parsed = response_model.model_validate(result.data)
                trace = CallTrace(
                    stage=stage,
                    model_name=result.model_name,
                    model_revision=result.model_revision,
                    temperature=temperature,
                    seed=seed,
                    prompt_version=self.prompt_version,
                    prompt_hash=prompt_hash,
                    input_hash=input_hash,
                    raw_response_path=str(cache_path),
                    attempts=attempt + 1,
                    effective_seed=effective_seed,
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
                detail: dict[str, object] = {
                    "attempt": attempt + 1,
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "retryable": retryable,
                    "effective_seed": effective_seed,
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
                LOGGER.warning("VLM %s failed: %s", stage, message)
                if not retryable or attempt >= self.max_retries:
                    break
                time.sleep(min(2**attempt, 4))
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
            f"VLM stage {stage} failed after {len(errors)} attempt(s): {'; '.join(errors)}"
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


def _backend_identity(backend: VisionLanguageBackend) -> dict[str, object]:
    identity = getattr(backend, "cache_identity", None)
    if callable(identity):
        value = identity()
        if not isinstance(value, dict):
            raise TypeError("backend cache_identity() must return a dict")
        return value
    return {"backend_contract": type(backend).__name__}


def _is_retryable_error(exc: Exception) -> bool:
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
