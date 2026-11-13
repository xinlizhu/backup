from __future__ import annotations

import base64
import json
import math
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from cir_qdiag_train.on_policy_config import OnPolicyTeacherConfig
from cir_qdiag_train.on_policy_data import CIRRPositiveRecord


class HttpRerankerClient:
    """Small dependency-free client for the vLLM-compatible rerank endpoint."""

    def __init__(self, config: OnPolicyTeacherConfig) -> None:
        if config.backend != "http_reranker":
            raise ValueError("HttpRerankerClient requires backend=http_reranker")
        if config.endpoint_url is None or config.model_name is None:
            raise ValueError("HTTP reranker endpoint and model name are required")
        self.config = config
        self.endpoint_url = config.endpoint_url
        self.model_name = config.model_name

    def score_candidates(
        self,
        *,
        record: CIRRPositiveRecord,
        candidate_paths: list[Path],
    ) -> np.ndarray:
        if not candidate_paths:
            raise ValueError("reranker candidate list is empty")
        query = {
            "content": [
                self._image_part(record.reference_path),
                {"type": "text", "text": record.text},
            ]
        }
        documents = [
            {"content": [self._image_part(path)]} for path in candidate_paths
        ]
        payload: dict[str, Any] = {
            "model": self.model_name,
            "documents": documents,
        }
        if self.config.api_format == "score":
            payload["queries"] = query
        else:
            payload["query"] = query
            payload["top_n"] = len(candidate_paths)
        if self.config.instruction is not None:
            payload["instruction"] = self.config.instruction
        response = self._post_json(payload, pair_id=record.pair_id)
        parser = (
            _parse_score_scores
            if self.config.api_format == "score"
            else _parse_rerank_scores
        )
        return parser(response, expected_count=len(candidate_paths), pair_id=record.pair_id)

    def _image_part(self, path: Path) -> dict[str, Any]:
        resolved = path.resolve()
        if self.config.image_transport == "file_url":
            url = resolved.as_uri()
        else:
            payload = resolved.read_bytes()
            mime_type = _detect_image_mime(payload, resolved)
            encoded = base64.b64encode(payload).decode("ascii")
            url = f"data:{mime_type};base64,{encoded}"
        return {"type": "image_url", "image_url": {"url": url}}

    def _post_json(self, payload: dict[str, Any], *, pair_id: int) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            request = Request(
                self.endpoint_url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlopen(
                    request, timeout=self.config.request_timeout_seconds
                ) as response:
                    response_body = response.read()
                parsed = json.loads(response_body)
                if not isinstance(parsed, dict):
                    raise TypeError("reranker response is not a JSON object")
                if "error" in parsed:
                    raise RuntimeError(f"reranker returned an error: {parsed['error']}")
                return parsed
            except HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")[:1000]
                last_error = RuntimeError(
                    f"HTTP {error.code} from reranker for pair {pair_id}: {detail}"
                )
                if error.code < 500 and error.code != 429:
                    break
            except (URLError, TimeoutError, json.JSONDecodeError, OSError) as error:
                last_error = error
            except (TypeError, RuntimeError) as error:
                last_error = error
                break
            if attempt < self.config.max_retries:
                delay = min(
                    30.0,
                    self.config.retry_backoff_seconds * (2**attempt),
                )
                time.sleep(delay)
        raise RuntimeError(
            f"reranker request failed for CIRR pair {pair_id} after "
            f"{self.config.max_retries + 1} attempts"
        ) from last_error


def _parse_rerank_scores(
    response: dict[str, Any],
    *,
    expected_count: int,
    pair_id: int,
) -> np.ndarray:
    results = response.get("results")
    if not isinstance(results, list) or len(results) != expected_count:
        raise ValueError(
            f"reranker returned {len(results) if isinstance(results, list) else 'invalid'} "
            f"results for {expected_count} candidates at pair {pair_id}"
        )
    scores = np.full(expected_count, np.nan, dtype=np.float32)
    for item in results:
        if not isinstance(item, dict):
            raise TypeError(f"reranker result is not an object at pair {pair_id}")
        index = item.get("index")
        score = item.get("relevance_score")
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError(f"reranker result has an invalid index at pair {pair_id}")
        if index < 0 or index >= expected_count or math.isfinite(float(scores[index])):
            raise ValueError(f"reranker result index is invalid or duplicated: {index}")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise TypeError(f"reranker result has an invalid score at pair {pair_id}")
        numeric_score = float(score)
        if not math.isfinite(numeric_score) or not 0.0 <= numeric_score <= 1.0:
            raise ValueError(
                f"reranker relevance_score must be finite and in [0, 1], got "
                f"{numeric_score} at pair {pair_id}"
            )
        scores[index] = numeric_score
    if not bool(np.isfinite(scores).all()):
        raise ValueError(f"reranker response omitted candidate scores at pair {pair_id}")
    return scores


def _parse_score_scores(
    response: dict[str, Any],
    *,
    expected_count: int,
    pair_id: int,
) -> np.ndarray:
    data = response.get("data")
    if not isinstance(data, list) or len(data) != expected_count:
        raise ValueError(
            f"score endpoint returned {len(data) if isinstance(data, list) else 'invalid'} "
            f"results for {expected_count} candidates at pair {pair_id}"
        )
    translated = {
        "results": [
            {
                "index": item.get("index") if isinstance(item, dict) else None,
                "relevance_score": item.get("score") if isinstance(item, dict) else None,
            }
            for item in data
        ]
    }
    return _parse_rerank_scores(
        translated,
        expected_count=expected_count,
        pair_id=pair_id,
    )


def _detect_image_mime(payload: bytes, path: Path) -> str:
    if payload.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if payload.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(payload) >= 12 and payload[:4] == b"RIFF" and payload[8:12] == b"WEBP":
        return "image/webp"
    suffix = path.suffix.lower()
    fallback = {
        ".bmp": "image/bmp",
        ".jpeg": "image/jpeg",
        ".jpg": "image/jpeg",
        ".png": "image/png",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".webp": "image/webp",
    }.get(suffix)
    if fallback is None:
        raise ValueError(f"unsupported reranker image format: {path}")
    return fallback
