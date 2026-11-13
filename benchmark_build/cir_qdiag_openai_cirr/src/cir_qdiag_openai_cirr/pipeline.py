from __future__ import annotations

import json
import logging
import os
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, Iterator, Literal, TypeVar

from cir_qdiag_openai_cirr.config import PipelineConfig
from cir_qdiag_openai_cirr.data import load_cirr_train
from cir_qdiag_openai_cirr.generation.direct import generate_operation
from cir_qdiag_openai_cirr.models.base import (
    FatalProviderError,
    StructuredLanguageBackend,
)
from cir_qdiag_openai_cirr.models.client import StructuredLLMClient
from cir_qdiag_openai_cirr.models.openai_chat import OpenAIChatBackend
from cir_qdiag_openai_cirr.schemas import (
    SCHEMA_VERSION,
    CIRRSampleRecord,
    EditOperation,
    NegativeCandidate,
)
from cir_qdiag_openai_cirr.utils.hashing import sha256_file, sha256_text
from cir_qdiag_openai_cirr.utils.jsonl import (
    ResumableJsonlWriter,
    read_models,
    write_jsonl_atomic,
)
from cir_qdiag_openai_cirr.validation import review_sample_candidates


LOGGER = logging.getLogger(__name__)
T = TypeVar("T")


def create_backend(config: PipelineConfig) -> StructuredLanguageBackend:
    model = config.model
    if model.backend == "openai":
        return OpenAIChatBackend(
            api_base=model.api_base,
            api_key=model.api_key,
            organization=model.organization,
            project=model.project,
            timeout_seconds=model.timeout_seconds,
            generation_model=model.generation_model,
            validation_model=model.validation_model,
            structured_output_mode=model.structured_output_mode,
            generation_reasoning_effort=model.generation_reasoning_effort,
            validation_reasoning_effort=model.validation_reasoning_effort,
            image_detail=model.image_detail,
        )
    raise ValueError(f"unsupported model backend: {model.backend}")


def create_client(
    config: PipelineConfig,
    backend: StructuredLanguageBackend | None = None,
) -> StructuredLLMClient:
    return StructuredLLMClient(
        backend=backend or create_backend(config),
        cache_dir=config.runtime.cache_dir,
        prompt_version=config.runtime.prompt_version,
        max_retries=config.model.max_retries,
    )


def prepare_source(
    config: PipelineConfig,
    output_path: str | Path | None = None,
    *,
    start_index: int | None = None,
    end_index: int | None = None,
) -> Path:
    """Build CIRR train source records locally without making API calls."""

    _validate_slice(start_index, end_index)
    output = Path(output_path or config.runtime.output_dir / "source.jsonl")
    records = load_cirr_train(
        config.data.annotation_path,
        config.data.image_split_path,
        config.data.image_root,
        verify_images=config.data.verify_images,
    )
    if config.sampling.strategy == "random":
        random.Random(config.sampling.seed).shuffle(records)
    if config.sampling.limit is not None:
        records = records[: config.sampling.limit]
    records = records[slice(start_index, end_index)]

    source_fingerprint = _source_fingerprint(config)
    selection_fingerprint = _selection_fingerprint(config, start_index, end_index)
    for record in records:
        record.metadata.stage = "source"
        record.metadata.operational_status = "ok"
        record.metadata.semantic_status = "pending"
        record.metadata.prompt_version = config.runtime.prompt_version
        record.metadata.build_fingerprint = source_fingerprint
        record.metadata.selection_fingerprint = selection_fingerprint
        record.metadata.random_seed = config.sampling.seed
        record.metadata.notes = [
            "GPT generation is text-only; validation receives Reference and Target images."
        ]
    write_jsonl_atomic(output, records)
    LOGGER.info("prepared %d CIRR train samples at %s", len(records), output)
    return output


def generate_stage(
    config: PipelineConfig,
    input_path: str | Path,
    output_path: str | Path,
    client: StructuredLLMClient | None = None,
    *,
    force: bool = False,
) -> Path:
    """Run ADD, DELETE, and SUBSTITUTE once each per CIRR sample."""

    client = client or create_client(config)
    output = Path(output_path)
    source_fingerprint = _source_fingerprint(config)
    build_fingerprint = _generation_fingerprint(config)
    input_records = list(read_models(input_path, CIRRSampleRecord))
    _require_stage(input_records, "source", "prepare")
    _require_fingerprint(input_records, source_fingerprint, "prepare")
    stage_abort = _ProviderStageAbort("generation")

    with ResumableJsonlWriter(
        output,
        id_field="sample_id",
        force=force,
        order_ids=[record.sample_id for record in input_records],
        retry_predicate=_retry_predicate(config.runtime.prompt_version, build_fingerprint),
    ) as writer:
        records = [record for record in input_records if not writer.has(record.sample_id)]

        def process(record: CIRRSampleRecord) -> CIRRSampleRecord:
            stage_abort.raise_if_set()
            result = record.model_copy(deep=True)
            result.metadata.stage = "candidates"
            result.metadata.prompt_version = config.runtime.prompt_version
            result.metadata.build_fingerprint = build_fingerprint
            result.metadata.generator_model = config.model.generation_model
            result.metadata.call_traces = []
            result.metadata.errors = []
            result.candidates = []
            result.negatives = []
            failed = False

            for operation in EditOperation:
                stage_abort.raise_if_set()
                try:
                    candidates, trace = generate_operation(
                        result,
                        operation,
                        client,
                        config.runtime.prompt_dir,
                        config.runtime.prompt_version,
                        config.generation.max_output_tokens,
                    )
                    result.metadata.call_traces.append(trace)
                    result.candidates.extend(candidates)
                except FatalProviderError as exc:
                    stage_abort.set(exc)
                    raise
                except Exception as exc:
                    failed = True
                    message = (
                        f"generate_{operation.value}:{type(exc).__name__}:{exc}"
                    )
                    result.metadata.errors.append(message)
                    LOGGER.exception(
                        "%s generation failed for %s",
                        operation.value,
                        result.sample_id,
                    )

            result.metadata.num_generated_candidates = len(result.candidates)
            result.metadata.num_accepted_negatives = 0
            result.metadata.operational_status = "failed" if failed else "ok"
            result.metadata.semantic_status = (
                "pending" if result.candidates else "rejected"
            )
            return CIRRSampleRecord.model_validate(result.model_dump(mode="json"))

        for result in _ordered_parallel(records, process, _stage_workers(config)):
            writer.append(result)
    return output


def validate_stage(
    config: PipelineConfig,
    input_path: str | Path,
    output_path: str | Path,
    client: StructuredLLMClient | None = None,
    *,
    force: bool = False,
) -> Path:
    """Review all candidates in one multimodal call with Reference and Target."""

    client = client or create_client(config)
    output = Path(output_path)
    generation_fingerprint = _generation_fingerprint(config)
    build_fingerprint = _validation_fingerprint(config)
    input_records = list(read_models(input_path, CIRRSampleRecord))
    _require_stage(input_records, "candidates", "generate")
    if not force:
        _require_fingerprint(input_records, generation_fingerprint, "generate")
    stage_abort = _ProviderStageAbort("validation")

    with ResumableJsonlWriter(
        output,
        id_field="sample_id",
        force=force,
        order_ids=[record.sample_id for record in input_records],
        retry_predicate=_retry_predicate(config.runtime.prompt_version, build_fingerprint),
    ) as writer:
        records = [record for record in input_records if not writer.has(record.sample_id)]

        def process(record: CIRRSampleRecord) -> CIRRSampleRecord:
            stage_abort.raise_if_set()
            result = record.model_copy(deep=True)
            result.metadata.stage = "final"
            result.metadata.prompt_version = config.runtime.prompt_version
            result.metadata.build_fingerprint = build_fingerprint
            result.metadata.validator_model = config.model.validation_model
            result.negatives = []
            result.metadata.num_accepted_negatives = 0

            if not result.candidates:
                result.metadata.semantic_status = "rejected"
                return CIRRSampleRecord.model_validate(result.model_dump(mode="json"))

            try:
                reviewed, traces = review_sample_candidates(
                    result,
                    client,
                    config.runtime.prompt_dir,
                    config.runtime.prompt_version,
                    config.validation.max_output_tokens,
                )
                result.metadata.call_traces.extend(traces)
                result.candidates = reviewed
                result.negatives = [
                    candidate
                    for candidate in reviewed
                    if candidate.status == "accepted"
                ]
                result.metadata.num_accepted_negatives = len(result.negatives)
                result.metadata.operational_status = (
                    "failed"
                    if record.metadata.operational_status == "failed"
                    else "ok"
                )
                result.metadata.semantic_status = _semantic_status(reviewed)
            except FatalProviderError as exc:
                stage_abort.set(exc)
                raise
            except Exception as exc:
                message = f"validate_visual:{type(exc).__name__}:{exc}"
                result.metadata.errors.append(message)
                LOGGER.exception("visual validation failed for %s", result.sample_id)
                for candidate in result.candidates:
                    candidate.status = "failed"
                    candidate.selection_reason = message
                result.metadata.operational_status = "failed"
                result.metadata.semantic_status = "pending"
            return CIRRSampleRecord.model_validate(result.model_dump(mode="json"))

        for result in _ordered_parallel(records, process, _stage_workers(config)):
            writer.append(result)
    return output


def run_pipeline(
    config: PipelineConfig,
    *,
    start_index: int | None = None,
    end_index: int | None = None,
    backend: StructuredLanguageBackend | None = None,
) -> dict[str, Path]:
    """Prepare CIRR train, generate text-only candidates, then review them visually."""

    _validate_slice(start_index, end_index)
    output_dir = config.runtime.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "source": output_dir / "source.jsonl",
        "candidates": output_dir / "candidates.jsonl",
        "final": output_dir / "final.jsonl",
    }
    force = config.runtime.force_stage
    if not config.runtime.resume and force is None:
        force = "prepare"
    force_order = {None: 99, "prepare": 0, "generate": 1, "validate": 2}
    force_index = force_order[force]
    source_fingerprint = _source_fingerprint(config)
    selection_fingerprint = _selection_fingerprint(config, start_index, end_index)
    source_is_stale = True
    if paths["source"].exists():
        source_is_stale = any(
            record.metadata.build_fingerprint != source_fingerprint
            or record.metadata.selection_fingerprint != selection_fingerprint
            for record in read_models(paths["source"], CIRRSampleRecord)
        )
    if force == "prepare" or not paths["source"].exists() or source_is_stale:
        prepare_source(
            config,
            paths["source"],
            start_index=start_index,
            end_index=end_index,
        )
        force_index = 0

    client = create_client(config, backend=backend)
    generate_stage(
        config,
        paths["source"],
        paths["candidates"],
        client,
        force=force_index <= 1,
    )
    validate_stage(
        config,
        paths["candidates"],
        paths["final"],
        client,
        force=force_index <= 2,
    )
    return paths


def _semantic_status(
    candidates: list[NegativeCandidate],
) -> Literal["pending", "accepted", "partial", "rejected"]:
    accepted = sum(item.status == "accepted" for item in candidates)
    if accepted == len(candidates) and candidates:
        return "accepted"
    if accepted:
        return "partial"
    if any(item.status == "failed" for item in candidates):
        return "pending"
    return "rejected"


def _retry_predicate(prompt_version: str, build_fingerprint: str):
    def should_retry(row: dict) -> bool:
        metadata = row.get("metadata", {})
        return bool(
            row.get("schema_version") != SCHEMA_VERSION
            or metadata.get("prompt_version") != prompt_version
            or metadata.get("build_fingerprint") != build_fingerprint
            or metadata.get("operational_status") == "failed"
        )

    return should_retry


def _source_fingerprint(config: PipelineConfig) -> str:
    payload = {
        "schema": SCHEMA_VERSION,
        "data": config.data.model_dump(mode="json"),
        "annotation_hash": sha256_file(config.data.annotation_path),
        "image_split_hash": sha256_file(config.data.image_split_path),
        "loader": _source_text("data/cirr_loader.py"),
        "schemas": _source_text("schemas.py"),
    }
    return _hash_payload(payload)


def _generation_fingerprint(config: PipelineConfig) -> str:
    payload = {
        "source": _source_fingerprint(config),
        "model": {
            "backend": config.model.backend,
            "api_base": _effective_api_base(config.model.api_base),
            "generation_model": config.model.generation_model,
            "structured_output_mode": config.model.structured_output_mode,
            "generation_reasoning_effort": config.model.generation_reasoning_effort,
        },
        "generation": config.generation.model_dump(mode="json"),
        "direct": _source_text("generation/direct.py"),
        "prompts": {
            operation.value: _prompt_text(config, f"generate_{operation.value}")
            for operation in EditOperation
        },
    }
    return _hash_payload(payload)


def _validation_fingerprint(config: PipelineConfig) -> str:
    payload = {
        "generation_schema": SCHEMA_VERSION,
        "model": {
            "backend": config.model.backend,
            "api_base": _effective_api_base(config.model.api_base),
            "validation_model": config.model.validation_model,
            "structured_output_mode": config.model.structured_output_mode,
            "validation_reasoning_effort": config.model.validation_reasoning_effort,
            "image_detail": config.model.image_detail,
        },
        "validation": config.validation.model_dump(mode="json"),
        "review": _source_text("validation/batch_review.py"),
        "prompt": _prompt_text(config, "validate_visual"),
    }
    return _hash_payload(payload)


def _selection_fingerprint(
    config: PipelineConfig,
    start_index: int | None,
    end_index: int | None,
) -> str:
    payload = {
        "sampling": config.sampling.model_dump(mode="json"),
        "start_index": start_index,
        "end_index": end_index,
    }
    return _hash_payload(payload)


def _source_text(relative: str) -> str:
    return (Path(__file__).resolve().parent / relative).read_text(encoding="utf-8")


def _prompt_text(config: PipelineConfig, stem: str) -> str:
    path = Path(config.runtime.prompt_dir) / (
        f"{stem}_{config.runtime.prompt_version}.txt"
    )
    return path.read_text(encoding="utf-8")


def _hash_payload(payload: object) -> str:
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _effective_api_base(value: str) -> str:
    match = re.fullmatch(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", value.strip())
    return os.environ.get(match.group(1), value) if match else value


def _require_fingerprint(
    records: list[CIRRSampleRecord],
    fingerprint: str,
    upstream_stage: str,
) -> None:
    stale = [
        record.sample_id
        for record in records
        if record.metadata.build_fingerprint != fingerprint
    ]
    if stale:
        preview = ", ".join(stale[:5])
        raise ValueError(
            f"{upstream_stage} output uses stale logic for {len(stale)} samples "
            f"({preview}); rerun that stage"
        )


def _require_stage(
    records: list[CIRRSampleRecord],
    expected_stage: str,
    upstream_stage: str,
) -> None:
    stale = [
        record.sample_id
        for record in records
        if record.metadata.stage != expected_stage
    ]
    if stale:
        preview = ", ".join(stale[:5])
        raise ValueError(
            f"{upstream_stage} output has the wrong stage for {len(stale)} samples "
            f"({preview}); expected metadata.stage={expected_stage!r}"
        )


def _validate_slice(start_index: int | None, end_index: int | None) -> None:
    if start_index is not None and start_index < 0:
        raise ValueError("start_index must be non-negative")
    if end_index is not None and end_index < 0:
        raise ValueError("end_index must be non-negative")
    if start_index is not None and end_index is not None and end_index < start_index:
        raise ValueError("end_index must be >= start_index")


def _stage_workers(config: PipelineConfig) -> int:
    return config.runtime.max_workers


class _ProviderStageAbort:
    """Share a provider-wide fatal failure across concurrent stage workers."""

    def __init__(self, stage: str) -> None:
        self.stage = stage
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._error: FatalProviderError | None = None

    def set(self, error: FatalProviderError) -> None:
        with self._lock:
            if self._error is None:
                self._error = error
            self._event.set()

    def raise_if_set(self) -> None:
        if not self._event.is_set():
            return
        with self._lock:
            error = self._error
        message = f"{self.stage} aborted after a provider-wide account/quota failure"
        if error is None:
            raise FatalProviderError(message)
        raise FatalProviderError(f"{message}: {error}") from error


def _ordered_parallel(
    items: Iterable[T],
    function: Callable[[T], T],
    max_workers: int,
) -> Iterator[T]:
    if max_workers <= 1:
        yield from map(function, items)
        return
    with ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="cirr-qdiag",
    ) as executor:
        yield from executor.map(function, items)
