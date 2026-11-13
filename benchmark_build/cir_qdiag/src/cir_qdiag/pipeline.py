from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, Iterator, Literal, TypeVar

from tqdm.auto import tqdm

from cir_qdiag.config import PipelineConfig
from cir_qdiag.data.cirr_loader import load_cirr_annotations
from cir_qdiag.data.sampling import sample_records
from cir_qdiag.generation.direct import generate_operation
from cir_qdiag.models.base import VisionLanguageBackend
from cir_qdiag.models.client import StructuredVLMClient
from cir_qdiag.models.transformers_vlm import TransformersVLMBackend
from cir_qdiag.models.vllm_openai import VLLMOpenAIBackend
from cir_qdiag.screening import screen_cirr_triplet
from cir_qdiag.schemas import (
    SCHEMA_VERSION,
    ConstructionMode,
    EditOperation,
    NegativeCandidate,
    SampleRecord,
)
from cir_qdiag.utils.hashing import (
    deterministic_seed,
    local_model_fingerprint,
    sha256_file,
    sha256_text,
)
from cir_qdiag.utils.jsonl import ResumableJsonlWriter, read_models, write_jsonl_atomic
from cir_qdiag.validation.batch_review import review_candidates


LOGGER = logging.getLogger(__name__)
T = TypeVar("T")


def create_backend(config: PipelineConfig) -> VisionLanguageBackend:
    model = config.model
    if model.backend == "vllm":
        return VLLMOpenAIBackend(
            model_name=model.name,
            model_source=str(model.name_or_path),
            api_base=model.api_base,
            api_key=model.api_key,
            timeout_seconds=model.timeout_seconds,
            model_revision=model.model_revision,
            disable_thinking=model.disable_thinking,
            mm_processor_kwargs=model.mm_processor_kwargs,
        )
    if model.backend == "transformers":
        return TransformersVLMBackend(
            name_or_path=model.name_or_path,
            device=model.device,
            dtype=model.dtype,
            trust_remote_code=model.trust_remote_code,
            model_revision=model.model_revision,
            mm_processor_kwargs=model.mm_processor_kwargs,
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
    _validate_slice(start_index, end_index)
    output = Path(output_path or config.runtime.output_dir / "source.jsonl")
    records = load_cirr_annotations(
        annotation_path=config.data.annotation_path,
        image_root=config.data.image_root,
        split=config.data.split,
        image_split_path=config.data.image_split_path,
    )
    selected = sample_records(
        records,
        limit=config.sampling.limit,
        seed=config.sampling.seed,
        strategy=config.sampling.strategy,
        random_count=config.sampling.random_count,
        targeted_count=config.sampling.targeted_count,
    )[slice(start_index, end_index)]
    build_fingerprint = _build_fingerprint(config)
    selection_fingerprint = _selection_fingerprint(config, start_index, end_index)
    for record in selected:
        record.metadata.stage = "source"
        record.metadata.operational_status = "ok"
        record.metadata.semantic_status = "pending"
        record.metadata.prompt_version = config.runtime.prompt_version
        record.metadata.build_fingerprint = build_fingerprint
        record.metadata.selection_fingerprint = selection_fingerprint
        record.metadata.random_seed = config.sampling.seed
    write_jsonl_atomic(output, selected)
    LOGGER.info(
        "prepared %d CIRR %s samples at %s",
        len(selected),
        config.data.split,
        output,
    )
    return output


def screen_source_stage(
    config: PipelineConfig,
    input_path: str | Path,
    output_path: str | Path,
    rejected_path: str | Path,
    client: StructuredVLMClient | None = None,
    *,
    failed_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Remove fundamentally invalid CIRR triplets before negative generation."""

    client = client or create_client(config)
    output = Path(output_path)
    rejected = Path(rejected_path)
    failed_output = Path(
        failed_path or rejected.with_name("source_failures.jsonl")
    )
    build_fingerprint = _build_fingerprint(config)
    input_records = list(read_models(input_path, SampleRecord))
    _require_prompt_version(input_records, config.runtime.prompt_version, "prepare")
    _require_build_fingerprint(input_records, build_fingerprint, "prepare")

    def process(record: SampleRecord) -> SampleRecord:
        result = record.model_copy(deep=True)
        result.metadata.stage = "screened"
        result.metadata.prompt_version = config.runtime.prompt_version
        result.metadata.build_fingerprint = build_fingerprint
        try:
            response, trace = screen_cirr_triplet(
                record,
                client,
                config.runtime.prompt_dir,
                config.runtime.prompt_version,
                config.screening.temperature,
                deterministic_seed(config.sampling.seed, record.sample_id, "screen_cirr"),
                config.screening.max_new_tokens,
            )
            result.metadata.source_screen_decision = response.decision
            result.metadata.source_screen_reason = response.reason
            result.metadata.call_traces.append(trace)
            result.metadata.operational_status = "ok"
            result.metadata.semantic_status = (
                "pending" if response.decision == "keep" else "rejected"
            )
        except Exception as exc:
            message = f"screen_cirr:{type(exc).__name__}:{exc}"
            result.metadata.source_screen_decision = "pending"
            result.metadata.source_screen_reason = None
            result.metadata.operational_status = "failed"
            result.metadata.semantic_status = "pending"
            result.metadata.errors.append(message)
            LOGGER.exception("source screen failed for %s", record.sample_id)
        return SampleRecord.model_validate(result.model_dump(mode="json"))

    screened: list[SampleRecord] = []
    kept_count = 0
    dropped_count = 0
    with tqdm(
        _ordered_parallel(input_records, process, _stage_workers(config)),
        total=len(input_records),
        desc="Screen CIRR",
        unit="sample",
        dynamic_ncols=True,
    ) as progress:
        for result in progress:
            screened.append(result)
            if result.metadata.source_screen_decision == "keep":
                kept_count += 1
            else:
                dropped_count += 1
            progress.set_postfix(
                kept=kept_count,
                dropped=dropped_count,
                refresh=False,
            )
    kept = [
        record
        for record in screened
        if record.metadata.operational_status == "ok"
        and record.metadata.source_screen_decision == "keep"
    ]
    dropped = [
        record
        for record in screened
        if record.metadata.operational_status == "ok"
        and record.metadata.source_screen_decision == "drop"
    ]
    failures = [
        record
        for record in screened
        if record.metadata.operational_status == "failed"
    ]
    write_jsonl_atomic(output, kept)
    write_jsonl_atomic(rejected, dropped)
    write_jsonl_atomic(failed_output, failures)
    LOGGER.info(
        "CIRR source screen kept %d, dropped %d, and isolated %d operational "
        "failures; outputs: %s, %s, %s",
        len(kept),
        len(dropped),
        len(failures),
        output,
        rejected,
        failed_output,
    )
    return output, rejected


def generate_stage(
    config: PipelineConfig,
    input_path: str | Path,
    output_path: str | Path,
    client: StructuredVLMClient | None = None,
    *,
    force: bool = False,
) -> Path:
    """Run exactly six direct-generation routes per sample."""

    client = client or create_client(config)
    output = Path(output_path)
    build_fingerprint = _build_fingerprint(config)
    input_records = list(read_models(input_path, SampleRecord))
    _require_prompt_version(input_records, config.runtime.prompt_version, "screen")
    _require_build_fingerprint(input_records, build_fingerprint, "screen")
    _require_stage(input_records, "screened", "screen")
    with ResumableJsonlWriter(
        output,
        force=force,
        order_ids=[record.sample_id for record in input_records],
        retry_predicate=_retry_predicate(config.runtime.prompt_version, build_fingerprint),
    ) as writer:
        records = [record for record in input_records if not writer.has(record.sample_id)]

        def process(record: SampleRecord) -> SampleRecord:
            result = record.model_copy(deep=True)
            result.metadata.stage = "candidates"
            result.metadata.prompt_version = config.runtime.prompt_version
            result.metadata.build_fingerprint = build_fingerprint
            result.candidates = []
            result.negatives = []
            result.metadata.num_generated_candidates = 0
            result.metadata.num_accepted_negatives = 0
            result.metadata.generator_model = config.model.name

            failed = False
            base_seed = deterministic_seed(
                config.sampling.seed, record.sample_id, "generate_v5"
            )
            for mode_index, mode in enumerate(ConstructionMode):
                for operation_index, operation in enumerate(EditOperation):
                    try:
                        candidates, trace = generate_operation(
                            result,
                            mode,
                            operation,
                            client,
                            config.runtime.prompt_dir,
                            config.runtime.prompt_version,
                            config.generation.temperature,
                            base_seed + mode_index * 1000 + operation_index * 101,
                            config.generation.max_new_tokens,
                        )
                        result.metadata.call_traces.append(trace)
                        result.candidates.extend(candidates)
                    except Exception as exc:
                        failed = True
                        message = (
                            f"generate_{mode.value}_{operation.value}:"
                            f"{type(exc).__name__}:{exc}"
                        )
                        result.metadata.errors.append(message)
                        LOGGER.exception(
                            "%s/%s generation failed for %s",
                            mode.value,
                            operation.value,
                            record.sample_id,
                        )

            result.candidates = _deduplicate_candidates(result.candidates, result)
            result.metadata.num_generated_candidates = len(result.candidates)
            result.metadata.operational_status = "failed" if failed else "ok"
            result.metadata.semantic_status = (
                "pending" if result.candidates else "rejected"
            )
            return SampleRecord.model_validate(result.model_dump(mode="json"))

        completed = len(input_records) - len(records)
        with tqdm(
            _ordered_parallel(records, process, _stage_workers(config)),
            total=len(input_records),
            initial=completed,
            desc="Generate negatives",
            unit="sample",
            dynamic_ncols=True,
        ) as progress:
            for result in progress:
                writer.append(result)
                progress.set_postfix(
                    candidates=result.metadata.num_generated_candidates,
                    status=result.metadata.operational_status,
                    refresh=False,
                )
    return output


def validate_stage(
    config: PipelineConfig,
    input_path: str | Path,
    output_path: str | Path,
    client: StructuredVLMClient | None = None,
    *,
    force: bool = False,
) -> Path:
    """Select final negatives with separate ADD/SUBSTITUTE and DELETE reviews."""

    client = client or create_client(config)
    output = Path(output_path)
    build_fingerprint = _build_fingerprint(config)
    input_records = list(read_models(input_path, SampleRecord))
    _require_prompt_version(input_records, config.runtime.prompt_version, "generate")
    _require_build_fingerprint(input_records, build_fingerprint, "generate")
    with ResumableJsonlWriter(
        output,
        force=force,
        order_ids=[record.sample_id for record in input_records],
        retry_predicate=_retry_predicate(config.runtime.prompt_version, build_fingerprint),
    ) as writer:
        records = [record for record in input_records if not writer.has(record.sample_id)]

        def process(record: SampleRecord) -> SampleRecord:
            result = record.model_copy(deep=True)
            result.metadata.stage = "final"
            result.metadata.prompt_version = config.runtime.prompt_version
            result.metadata.build_fingerprint = build_fingerprint
            result.negatives = []
            result.metadata.num_accepted_negatives = 0
            result.metadata.validator_model = config.model.name

            if not result.candidates:
                result.metadata.semantic_status = "rejected"
                return SampleRecord.model_validate(result.model_dump(mode="json"))

            try:
                reviewed, traces = review_candidates(
                    result,
                    client,
                    config.runtime.prompt_dir,
                    config.runtime.prompt_version,
                    config.selection.temperature,
                    deterministic_seed(
                        config.sampling.seed, record.sample_id, "select_v5"
                    ),
                    config.selection.max_new_tokens,
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
            except Exception as exc:
                message = f"select_candidates:{type(exc).__name__}:{exc}"
                result.metadata.errors.append(message)
                LOGGER.exception("batch selection failed for %s", record.sample_id)
                for candidate in result.candidates:
                    candidate.status = "failed"
                    candidate.selection_reason = message
                result.metadata.operational_status = "failed"
                result.metadata.semantic_status = "pending"
            return SampleRecord.model_validate(result.model_dump(mode="json"))

        completed = len(input_records) - len(records)
        with tqdm(
            _ordered_parallel(records, process, _stage_workers(config)),
            total=len(input_records),
            initial=completed,
            desc="Validate negatives",
            unit="sample",
            dynamic_ncols=True,
        ) as progress:
            for result in progress:
                writer.append(result)
                progress.set_postfix(
                    accepted=result.metadata.num_accepted_negatives,
                    status=result.metadata.operational_status,
                    refresh=False,
                )
    return output


def run_pipeline(
    config: PipelineConfig,
    *,
    start_index: int | None = None,
    end_index: int | None = None,
    backend: VisionLanguageBackend | None = None,
) -> dict[str, Path]:
    _validate_slice(start_index, end_index)
    output_dir = config.runtime.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "source": output_dir / "source.jsonl",
        "screened": output_dir / "screened.jsonl",
        "source_rejections": output_dir / "source_rejections.jsonl",
        "source_failures": output_dir / "source_failures.jsonl",
        "candidates": output_dir / "candidates.jsonl",
        "final": output_dir / "final.jsonl",
    }
    force = config.runtime.force_stage
    if not config.runtime.resume and force is None:
        force = "prepare"
    force_order = {None: 99, "prepare": 0, "generate": 1, "validate": 2}
    force_index = force_order[force]
    build_fingerprint = _build_fingerprint(config)
    selection_fingerprint = _selection_fingerprint(config, start_index, end_index)
    source_is_stale = False
    if paths["source"].exists():
        source_is_stale = any(
            record.metadata.build_fingerprint != build_fingerprint
            or record.metadata.selection_fingerprint != selection_fingerprint
            for record in read_models(paths["source"], SampleRecord)
        )
    source_rebuilt = force == "prepare" or not paths["source"].exists() or source_is_stale
    if source_rebuilt:
        prepare_source(
            config,
            paths["source"],
            start_index=start_index,
            end_index=end_index,
        )
        force_index = 0
    client = create_client(config, backend=backend)
    previous_screened_ids = (
        {record.sample_id for record in read_models(paths["screened"], SampleRecord)}
        if paths["screened"].exists()
        else set()
    )
    screen_source_stage(
        config,
        paths["source"],
        paths["screened"],
        paths["source_rejections"],
        client,
        failed_path=paths["source_failures"],
    )
    screened_ids = {
        record.sample_id for record in read_models(paths["screened"], SampleRecord)
    }
    if screened_ids != previous_screened_ids:
        force_index = min(force_index, 1)
    generate_stage(
        config,
        paths["screened"],
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


def _deduplicate_candidates(
    candidates: list[NegativeCandidate], record: SampleRecord
) -> list[NegativeCandidate]:
    unique: list[NegativeCandidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = " ".join(candidate.text.casefold().split())
        if normalized in seen:
            record.metadata.notes.append(
                f"dropped duplicate generated text: {candidate.negative_id}"
            )
            continue
        seen.add(normalized)
        unique.append(candidate)
    return unique


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


def _build_fingerprint(config: PipelineConfig) -> str:
    """Hash active source, prompts, configuration, and CIRR annotations."""

    config_payload = config.model_dump(mode="json", exclude={"runtime"})
    parts = [
        f"schema:{SCHEMA_VERSION}",
        f"prompt_version:{config.runtime.prompt_version}",
        "config:" + json.dumps(config_payload, ensure_ascii=False, sort_keys=True),
    ]
    package_root = Path(__file__).resolve().parent
    prompt_root = Path(config.runtime.prompt_dir).resolve()
    for path in sorted(package_root.rglob("*.py")):
        parts.append(f"source:{path.relative_to(package_root)}\n{path.read_text(encoding='utf-8')}")
    for path in sorted(prompt_root.glob(f"*_{config.runtime.prompt_version}.txt")):
        parts.append(f"prompt:{path.name}\n{path.read_text(encoding='utf-8')}")
    for label, path in (
        ("annotation", config.data.annotation_path),
        ("image_split", config.data.image_split_path),
    ):
        resolved = Path(path).resolve()
        digest = sha256_file(resolved) if resolved.is_file() else "missing"
        parts.append(f"data:{label}:{resolved}:{digest}")
    parts.append(
        "model_artifact:"
        + str(local_model_fingerprint(config.model.name_or_path) or "remote-or-missing")
    )
    return sha256_text("\n---\n".join(parts))


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
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _validate_slice(start_index: int | None, end_index: int | None) -> None:
    if start_index is not None and start_index < 0:
        raise ValueError("start_index must be non-negative")
    if end_index is not None and end_index < 0:
        raise ValueError("end_index must be non-negative")
    if start_index is not None and end_index is not None and end_index < start_index:
        raise ValueError("end_index must be greater than or equal to start_index")


def _require_prompt_version(
    records: list[SampleRecord],
    prompt_version: str,
    upstream_stage: str,
) -> None:
    stale = [
        record.sample_id for record in records if record.metadata.prompt_version != prompt_version
    ]
    if stale:
        preview = ", ".join(stale[:5])
        raise ValueError(
            f"{upstream_stage} output uses a stale prompt version for {len(stale)} records "
            f"({preview}); rerun that stage with prompt_version={prompt_version}"
        )


def _require_build_fingerprint(
    records: list[SampleRecord],
    build_fingerprint: str,
    upstream_stage: str,
) -> None:
    stale = [
        record.sample_id
        for record in records
        if record.metadata.build_fingerprint != build_fingerprint
    ]
    if stale:
        preview = ", ".join(stale[:5])
        raise ValueError(
            f"{upstream_stage} output uses stale build logic for {len(stale)} records "
            f"({preview}); rerun that stage with the current code and prompts"
        )


def _require_stage(
    records: list[SampleRecord],
    expected_stage: str,
    upstream_stage: str,
) -> None:
    stale = [
        record.sample_id for record in records if record.metadata.stage != expected_stage
    ]
    if stale:
        preview = ", ".join(stale[:5])
        raise ValueError(
            f"{upstream_stage} output has the wrong stage for {len(stale)} records "
            f"({preview}); expected metadata.stage={expected_stage!r}"
        )


def _stage_workers(config: PipelineConfig) -> int:
    return 1 if config.model.backend == "transformers" else config.runtime.max_workers


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
        thread_name_prefix="cir-qdiag",
    ) as executor:
        yield from executor.map(function, items)
