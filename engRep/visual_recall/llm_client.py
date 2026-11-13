from __future__ import annotations

import base64
import json
import mimetypes
import math
import re
from pathlib import Path
from typing import Any

from .prompts import SYSTEM_PROMPT, build_user_text


MODEL_FIELDS = {
    "decision",
    "category",
    "filter_reason",
    "title",
    "image_content",
    "is_blurry",
    "confidence",
}


MODEL_JSON_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": sorted(MODEL_FIELDS),
    "properties": {
        "decision": {"type": "string", "enum": ["keep", "drop", "review"]},
        "category": {"type": "string", "enum": ["chart", "table", "diagram", "map", "photo", "screenshot", "logo", "icon", "decoration", "other"]},
        **{key: {"type": "string"} for key in ("filter_reason", "title", "image_content")},
        "is_blurry": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}

EVIDENCE_STATUSES = ('sufficient', 'missing_metric_labels', 'missing_period_labels', 'unreadable_key_text', 'conflicting_evidence')
RETRIEVAL_JSON_SCHEMA = {
    **MODEL_JSON_SCHEMA,
    'required': [*MODEL_JSON_SCHEMA['required'], 'evidence_status'],
    'properties': {**MODEL_JSON_SCHEMA['properties'], 'evidence_status': {'type': 'string', 'enum': list(EVIDENCE_STATUSES)}},
}


def validate_retrieval_output(value):
    if not isinstance(value, dict):
        return ['model output must be an object']
    errors = validate_model_output({k: v for k, v in value.items() if k != 'evidence_status'})
    if value.get('evidence_status') not in EVIDENCE_STATUSES:
        errors.append('missing or invalid evidence_status')
    return errors


class ModelOutputError(ValueError):
    def __init__(self, message, raw, metadata):
        super().__init__(message)
        self.raw = raw
        self.metadata = metadata


def image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def build_messages(payload: dict[str, Any], target_image: Path, page_image: Path | None,
                   *, system_prompt: str = SYSTEM_PROMPT, user_text: str | None = None) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {"type": "text", "text": "Image 1: high-resolution crop of the target Figure/Table (primary visual evidence)."},
        {"type": "image_url", "image_url": {"url": image_data_url(target_image)}},
    ]
    if page_image is not None and page_image.is_file():
        content.extend(
            [
                {"type": "text", "text": "Image 2: full report page with the current target outlined in red."},
                {"type": "image_url", "image_url": {"url": image_data_url(page_image)}},
            ]
        )
    content.append({"type": "text", "text": build_user_text(payload) if user_text is None else user_text})
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def parse_json_object(text: str) -> dict[str, Any]:
    clean = text.strip()
    clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s*```$", "", clean)
    try:
        value = json.loads(clean)
    except json.JSONDecodeError:
        start = clean.find("{")
        end = clean.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("model response does not contain a JSON object")
        value = json.loads(clean[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("model response JSON is not an object")
    return value


def validate_model_output(value: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, dict):
        return ['model output must be an object']
    missing = MODEL_FIELDS - set(value)
    if missing:
        errors.append(f"missing model fields: {sorted(missing)}")
    extra = set(value) - MODEL_FIELDS
    if extra:
        errors.append(f"unexpected model fields: {sorted(extra)}")
    if value.get("decision") not in {"keep", "drop", "review"}:
        errors.append("invalid decision")
    if value.get("category") not in {
        "chart", "table", "diagram", "map", "photo", "screenshot", "logo", "icon", "decoration", "other"
    }:
        errors.append("invalid category")
    if not isinstance(value.get("title", ""), str):
        errors.append("title must be a string")
    if not isinstance(value.get("image_content", ""), str):
        errors.append("image_content must be a string")
    if not isinstance(value.get('filter_reason'), str):
        errors.append('filter_reason must be a string')
    if not isinstance(value.get("is_blurry"), bool):
        errors.append("is_blurry must be boolean")
    confidence = value.get('confidence')
    if type(confidence) not in (int, float):
        errors.append("confidence must be numeric")
    elif not math.isfinite(confidence) or not 0 <= confidence <= 1:
        errors.append("confidence must be between 0 and 1")
    return errors


class QwenOpenAIClient:
    def __init__(
        self,
        *,
        api_base: str,
        api_key: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 1800,
        json_mode: bool = True,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai Python package is required; install requirements_visual.txt") from exc
        self.client = OpenAI(base_url=api_base.rstrip("/") + "/", api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.json_mode = json_mode

    def analyze(
        self,
        payload: dict[str, Any],
        target_image: Path,
        page_image: Path | None,
        *,
        system_prompt: str = SYSTEM_PROMPT,
        user_text: str | None = None,
        allow_json_fallback: bool = True,
        strict_schema: bool = False,
        retrieval_checks: bool = False,
    ) -> tuple[dict[str, Any], str, dict[str, Any]]:
        messages = build_messages(payload, target_image, page_image,
                                  system_prompt=system_prompt, user_text=user_text)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
        if self.json_mode:
            kwargs["response_format"] = (
                {"type": "json_schema", "json_schema": {"name": "visual_record", "strict": True, "schema": RETRIEVAL_JSON_SCHEMA if retrieval_checks else MODEL_JSON_SCHEMA}}
                if strict_schema else {"type": "json_object"}
            )
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception:
            if not self.json_mode or not allow_json_fallback:
                raise
            # Older vLLM builds may not expose response_format for this model.
            kwargs.pop("response_format", None)
            response = self.client.chat.completions.create(**kwargs)
        raw = response.choices[0].message.content or ""
        usage = response.usage.model_dump(exclude_none=True) if response.usage is not None else {}
        response_metadata = {
            "id": response.id,
            "model": response.model,
            "created": response.created,
            "finish_reason": response.choices[0].finish_reason,
            "usage": usage,
        }
        try:
            parsed = parse_json_object(raw)
            errors = validate_retrieval_output(parsed) if retrieval_checks else validate_model_output(parsed)
            if response.choices[0].finish_reason == "length":
                errors.append("response truncated by max_tokens")
            if errors:
                raise ValueError("; ".join(errors))
            parsed["confidence"] = float(parsed["confidence"])
        except (ValueError, TypeError) as exc:
            raise ModelOutputError(str(exc), raw, response_metadata) from exc
        return parsed, raw, response_metadata
