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

from cir_qdiag.config import PipelineConfig
from cir_qdiag.data import load_exact_pinpoint_groups
from cir_qdiag.generation.direct import generate_operation
from cir_qdiag.models.base import FatalProviderError, VisionLanguageBackend
from cir_qdiag.models.client import StructuredVLMClient
from cir_qdiag.models.openai_chat import OpenAIChatBackend
from cir_qdiag.schemas import (
    SCHEMA_VERSION,
    EditOperation,
    NegativeCandidate,
    PinPointGroupRecord,
)
from cir_qdiag.utils.hashing import sha256_file, sha256_text
from cir_qdiag.utils.jsonl import ResumableJsonlWriter, read_models, write_jsonl_atomic
from cir_qdiag.validation import (
    review_group_candidates,
    select_validation_positives,
)


LOGGER = logging.getLogger(__name__)
T = TypeVar("T")


def create_backend(config: PipelineConfig) -> VisionLanguageBackend:
    model = config.model
    if model.backend == "openai":
        return OpenAIChatBackend(
            api_base=model.api_base,
            api_key=model.api_key,
            organization=model.organization,
            project=model.project,
            timeout_seconds=model.timeout_seconds,
            image_detail=model.image_detail,
            generation_model=model.generation_model,
            selection_model=model.selection_model,
            structured_output_mode=model.structured_output_mode,
            generation_reasoning_effort=model.generation_reasoning_effort,
            selection_reasoning_effort=model.selection_reasoning_effort,
        )
    raise ValueError(f"unsupported model backend: {model.backend}")


def create_client(
    config: PipelineConfig,
    backend: VisionLanguageBackend | None = None,
) -> StructuredVLMClient:
    return StructuredVLMClient(
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
    """Build exact PinPoint groups locally; no model-based source screening."""

    _validate_slice(start_index, end_index)
    output = Path(output_path or config.runtime.output_dir / "groups.jsonl")
    records = load_exact_pinpoint_groups(
        config.data.parquet_path,
        config.data.image_root,
        min_group_size=config.data.min_group_size,
        max_group_size=config.data.max_group_size,
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
    write_jsonl_atomic(output, records)
    LOGGER.info(
        "prepared %d exact PinPoint paraphrase groups at %s",
        len(records),
        output,
    )
    return output


def generate_stage(
    config: PipelineConfig,
    input_path: str | Path,
    output_path: str | Path,
    client: StructuredVLMClient | None = None,
    *,
    force: bool = False,
) -> Path:
    """Run three text-only generation calls per instruction, yielding the 3x3 grid."""

    client = client or create_client(config)
    output = Path(output_path)
    source_fingerprint = _source_fingerprint(config)
    build_fingerprint = _generation_fingerprint(config)
    input_records = list(read_models(input_path, PinPointGroupRecord))
    _require_stage(input_records, "source", "prepare")
    _require_fingerprint(input_records, source_fingerprint, "prepare")
    stage_abort = _ProviderStageAbort("generation")

    with ResumableJsonlWriter(
        output,
        id_field="group_id",
        force=force,
        order_ids=[record.group_id for record in input_records],
        retry_predicate=_retry_predicate(config.runtime.prompt_version, build_fingerprint),
    ) as writer:
        records = [record for record in input_records if not writer.has(record.group_id)]

        def process(record: PinPointGroupRecord) -> PinPointGroupRecord:
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

            for instruction in result.instructions:
                for operation in EditOperation:
                    stage_abort.raise_if_set()
                    try:
                        candidates, trace = generate_operation(
                            result.group_id,
                            instruction,
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
                            f"{instruction.instruction_id}/generate_{operation.value}:"
                            f"{type(exc).__name__}:{exc}"
                        )
                        result.metadata.errors.append(message)
                        LOGGER.exception(
                            "%s generation failed for %s/%s",
                            operation.value,
                            result.group_id,
                            instruction.instruction_id,
                        )

            result.metadata.num_generated_candidates = len(result.candidates)
            result.metadata.num_accepted_negatives = 0
            result.metadata.operational_status = "failed" if failed else "ok"
            result.metadata.semantic_status = (
                "pending" if result.candidates else "rejected"
            )
            return PinPointGroupRecord.model_validate(result.model_dump(mode="json"))

        for result in _ordered_parallel(records, process, _stage_workers(config)):
            writer.append(result)
    return output


def validate_stage(
    config: PipelineConfig,
    input_path: str | Path,
    output_path: str | Path,
    client: StructuredVLMClient | None = None,
    *,
    force: bool = False,
) -> Path:
    """Review all candidates in one group-level multimodal call."""

    client = client or create_client(config)
    output = Path(output_path)
    generation_fingerprint = _generation_fingerprint(config)
    build_fingerprint = _validation_fingerprint(config)
    input_records = list(read_models(input_path, PinPointGroupRecord))
    _require_stage(input_records, "candidates", "generate")
    if not force:
        _require_fingerprint(input_records, generation_fingerprint, "generate")
    stage_abort = _ProviderStageAbort("validation")

    with ResumableJsonlWriter(
        output,
        id_field="group_id",
        force=force,
        order_ids=[record.group_id for record in input_records],
        retry_predicate=_retry_predicate(config.runtime.prompt_version, build_fingerprint),
    ) as writer:
        records = [record for record in input_records if not writer.has(record.group_id)]

        def process(record: PinPointGroupRecord) -> PinPointGroupRecord:
            stage_abort.raise_if_set()
            result = record.model_copy(deep=True)
            result.metadata.stage = "final"
            result.metadata.prompt_version = config.runtime.prompt_version
            result.metadata.build_fingerprint = build_fingerprint
            result.metadata.validator_model = config.model.selection_model
            result.negatives = []
            result.metadata.num_accepted_negatives = 0
            validation_positives = select_validation_positives(
                result,
                config.selection.max_positive_images,
            )
            result.metadata.validated_positive_image_ids = [
                item.image_id for item in validation_positives
            ]

            if not result.candidates:
                result.metadata.validated_positive_image_ids = []
                result.metadata.semantic_status = "rejected"
                return PinPointGroupRecord.model_validate(result.model_dump(mode="json"))

            try:
                stage_abort.raise_if_set()
                reviewed, traces = review_group_candidates(
                    result,
                    client,
                    config.runtime.prompt_dir,
                    config.runtime.prompt_version,
                    config.selection.max_output_tokens,
                    validation_positives,
                )
                result.metadata.call_traces.extend(traces)
                result.candidates = reviewed
                result.negatives = [
                    item for item in reviewed if item.status == "accepted"
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
                message = f"validate_group:{type(exc).__name__}:{exc}"
                result.metadata.errors.append(message)
                LOGGER.exception("group validation failed for %s", result.group_id)
                for candidate in result.candidates:
                    candidate.status = "failed"
                    candidate.selection_reason = message
                result.metadata.operational_status = "failed"
                result.metadata.semantic_status = "pending"
            return PinPointGroupRecord.model_validate(result.model_dump(mode="json"))

        for result in _ordered_parallel(records, process, _stage_workers(config)):
            writer.append(result)
    return output


def run_pipeline(
    config: PipelineConfig,
    *,
    start_index: int | None = None,
    end_index: int | None = None,
    backend: VisionLanguageBackend | None = None,
) -> dict[str, Path]:
    """Prepare local groups, generate 3x3 texts, then validate once per group."""

    _validate_slice(start_index, end_index)
    output_dir = config.runtime.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "source": output_dir / "groups.jsonl",
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
            for record in read_models(paths["source"], PinPointGroupRecord)
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
        "parquet_hash": sha256_file(config.data.parquet_path),
        "loader": _source_text("data/pinpoint_loader.py"),
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
            "selection_model": config.model.selection_model,
            "structured_output_mode": config.model.structured_output_mode,
            "selection_reasoning_effort": config.model.selection_reasoning_effort,
            "image_detail": config.model.image_detail,
        },
        "selection": config.selection.model_dump(mode="json"),
        "review": _source_text("validation/batch_review.py"),
        "prompt": _prompt_text(config, "validate_group"),
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
    records: list[PinPointGroupRecord],
    fingerprint: str,
    upstream_stage: str,
) -> None:
    stale = [
        record.group_id
        for record in records
        if record.metadata.build_fingerprint != fingerprint
    ]
    if stale:
        preview = ", ".join(stale[:5])
        raise ValueError(
            f"{upstream_stage} output uses stale logic for {len(stale)} groups "
            f"({preview}); rerun that stage"
        )


def _require_stage(
    records: list[PinPointGroupRecord],
    expected_stage: str,
    upstream_stage: str,
) -> None:
    stale = [
        record.group_id
        for record in records
        if record.metadata.stage != expected_stage
    ]
    if stale:
        preview = ", ".join(stale[:5])
        raise ValueError(
            f"{upstream_stage} output has the wrong stage for {len(stale)} groups "
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
        thread_name_prefix="pinpoint-qdiag",
    ) as executor:
        yield from executor.map(function, items)
