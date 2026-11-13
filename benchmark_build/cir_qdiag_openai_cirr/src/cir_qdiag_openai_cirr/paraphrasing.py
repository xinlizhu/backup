from __future__ import annotations

import csv
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean
from typing import Iterable, Iterator, Sequence, TypeVar

from tqdm.auto import tqdm

from cir_qdiag_openai_cirr.config import PipelineConfig
from cir_qdiag_openai_cirr.models.base import FatalProviderError
from cir_qdiag_openai_cirr.models.client import StructuredLLMClient
from cir_qdiag_openai_cirr.schemas import (
    PARAPHRASE_SCHEMA_VERSION,
    CIRRSampleRecord,
    CallTrace,
    ParaphraseGenerationResponse,
    ParaphraseMetrics,
    ParaphraseReviewResponse,
    PositiveParaphrase,
    PositiveParaphraseMetadata,
    PositiveParaphraseRecord,
)
from cir_qdiag_openai_cirr.utils.hashing import sha256_text
from cir_qdiag_openai_cirr.utils.jsonl import ResumableJsonlWriter, read_models
from cir_qdiag_openai_cirr.utils.prompts import dump_prompt_json, load_prompt


LOGGER = logging.getLogger(__name__)
T = TypeVar("T")


def positive_paraphrase_stage(
    config: PipelineConfig,
    input_path: str | Path,
    output_path: str | Path,
    client: StructuredLLMClient | None = None,
    *,
    force: bool = False,
    num_variants: int | None = None,
) -> Path:
    """Generate and textually verify positive paraphrases with OpenAI calls."""

    if client is None:
        from cir_qdiag_openai_cirr.pipeline import create_client

        client = create_client(config)
    requested = num_variants or config.paraphrase.num_variants
    if not 1 <= requested <= 20:
        raise ValueError("num_variants must be between 1 and 20")

    records = list(read_models(input_path, CIRRSampleRecord))
    output = Path(output_path)
    fingerprint = _build_fingerprint(config, requested)
    with ResumableJsonlWriter(
        output,
        force=force,
        order_ids=[record.sample_id for record in records],
        retry_predicate=_retry_predicate(
            prompt_version=config.runtime.prompt_version,
            build_fingerprint=fingerprint,
            requested=requested,
            generator_model=config.model.generation_model,
            reviewer_model=config.model.validation_model,
        ),
    ) as writer:
        pending = [record for record in records if not writer.has(record.sample_id)]

        def process(record: CIRRSampleRecord) -> PositiveParaphraseRecord:
            metadata = PositiveParaphraseMetadata(
                requested_variants=requested,
                generator_model=config.model.generation_model,
                reviewer_model=config.model.validation_model,
                prompt_version=config.runtime.prompt_version,
                build_fingerprint=fingerprint,
            )
            result = PositiveParaphraseRecord(
                sample_id=record.sample_id,
                pair_id=record.pair_id,
                reference=record.reference,
                target=record.target,
                positive_text=record.positive_text,
                metadata=metadata,
            )
            try:
                candidates, generation_trace = generate_positive_paraphrases(
                    record=record,
                    client=client,
                    prompt_dir=config.runtime.prompt_dir,
                    prompt_version=config.runtime.prompt_version,
                    requested=requested,
                    max_output_tokens=(
                        config.paraphrase.generation_max_output_tokens
                    ),
                )
                result.metadata.call_traces.append(generation_trace)
                result.candidates = candidates
                result.metadata.num_generated = len(candidates)
                if len(candidates) < requested:
                    result.metadata.notes.append(
                        f"generated {len(candidates)} unique rewrites "
                        f"out of {requested} requested"
                    )
                if not candidates:
                    raise RuntimeError(
                        "OpenAI returned no non-identical unique paraphrases"
                    )

                reviewed, review_traces = review_positive_paraphrases(
                    positive_text=record.positive_text,
                    candidates=candidates,
                    client=client,
                    prompt_dir=config.runtime.prompt_dir,
                    prompt_version=config.runtime.prompt_version,
                    max_output_tokens=config.paraphrase.review_max_output_tokens,
                )
                result.metadata.call_traces.extend(review_traces)
                result.candidates = reviewed
                result.paraphrases = [
                    candidate
                    for candidate in reviewed
                    if candidate.status == "accepted"
                ]
                result.metadata.num_accepted = len(result.paraphrases)
            except FatalProviderError:
                raise
            except Exception as exc:
                result.metadata.operational_status = "failed"
                result.metadata.errors.append(f"{type(exc).__name__}:{exc}")
                LOGGER.exception("positive paraphrasing failed for %s", record.sample_id)
            return PositiveParaphraseRecord.model_validate(
                result.model_dump(mode="json")
            )

        completed = len(records) - len(pending)
        results = _ordered_parallel(
            pending,
            process,
            config.paraphrase.max_workers,
        )
        with tqdm(
            results,
            total=len(records),
            initial=completed,
            desc="Positive paraphrases",
            unit="sample",
            dynamic_ncols=True,
        ) as progress:
            for result in progress:
                writer.append(result)
                progress.set_postfix(
                    accepted=result.metadata.num_accepted,
                    generated=result.metadata.num_generated,
                    status=result.metadata.operational_status,
                    refresh=False,
                )
    return output


def generate_positive_paraphrases(
    *,
    record: CIRRSampleRecord,
    client: StructuredLLMClient,
    prompt_dir: str | Path,
    prompt_version: str,
    requested: int,
    max_output_tokens: int,
) -> tuple[list[PositiveParaphrase], CallTrace]:
    prompt, _ = load_prompt(
        prompt_dir,
        "generate_positive_paraphrases",
        prompt_version,
    )
    response, trace = client.call(
        ParaphraseGenerationResponse,
        images=[],
        system_prompt=prompt,
        user_prompt=dump_prompt_json(
            {
                "positive_text": record.positive_text,
                "requested_variants": requested,
            }
        ),
        stage="generate_positive_paraphrases",
        max_output_tokens=max_output_tokens,
    )

    seen = {_normalize(record.positive_text)}
    candidates: list[PositiveParaphrase] = []
    for proposal in response.paraphrases:
        text = _clean_text(proposal.text)
        strategy = _clean_text(proposal.rewrite_strategy)
        normalized = _normalize(text)
        if not text or not strategy or normalized in seen:
            continue
        seen.add(normalized)
        candidates.append(
            PositiveParaphrase(
                paraphrase_id=f"{record.sample_id}_p{len(candidates) + 1}",
                text=text,
                rewrite_strategy=strategy,
                metrics=compute_paraphrase_metrics(record.positive_text, text),
            )
        )
        if len(candidates) == requested:
            break
    return candidates, trace


def review_positive_paraphrases(
    *,
    positive_text: str,
    candidates: list[PositiveParaphrase],
    client: StructuredLLMClient,
    prompt_dir: str | Path,
    prompt_version: str,
    max_output_tokens: int,
) -> tuple[list[PositiveParaphrase], list[CallTrace]]:
    prompt, _ = load_prompt(
        prompt_dir,
        "review_positive_paraphrases",
        prompt_version,
    )
    payload = {
        "positive_text": positive_text,
        "candidates": [
            {
                "paraphrase_id": candidate.paraphrase_id,
                "text": candidate.text,
                "rewrite_strategy": candidate.rewrite_strategy,
            }
            for candidate in candidates
        ],
    }
    traces: list[CallTrace] = []
    errors: list[str] = []
    for _ in range(client.max_retries + 1):
        response, trace = client.call(
            ParaphraseReviewResponse,
            images=[],
            system_prompt=prompt,
            user_prompt=dump_prompt_json(payload),
            stage="review_positive_paraphrases",
            max_output_tokens=max_output_tokens,
        )
        traces.append(trace)
        error = _review_partition_error(response, candidates)
        if error is None:
            by_id = {review.paraphrase_id: review for review in response.reviews}
            reviewed: list[PositiveParaphrase] = []
            for candidate in candidates:
                result = candidate.model_copy(deep=True)
                decision = by_id[candidate.paraphrase_id]
                result.status = (
                    "accepted"
                    if decision.decision == "equivalent"
                    else "rejected"
                )
                result.review_reason = decision.reason
                reviewed.append(result)
            return reviewed, traces
        errors.append(error)
        client.reject_cached_response(trace, error)
    raise RuntimeError(
        "review_positive_paraphrases returned an invalid partition: "
        + "; ".join(errors)
    )


def compute_paraphrase_metrics(
    source_text: str,
    paraphrase_text: str,
) -> ParaphraseMetrics:
    source = _clean_text(source_text)
    paraphrase = _clean_text(paraphrase_text)
    source_tokens = _tokens(source)
    paraphrase_tokens = _tokens(paraphrase)
    char_distance = _edit_distance(source, paraphrase)
    token_distance = _edit_distance(source_tokens, paraphrase_tokens)
    token_union = set(source_tokens) | set(paraphrase_tokens)
    token_intersection = set(source_tokens) & set(paraphrase_tokens)
    return ParaphraseMetrics(
        source_char_count=len(source),
        paraphrase_char_count=len(paraphrase),
        source_token_count=len(source_tokens),
        paraphrase_token_count=len(paraphrase_tokens),
        length_ratio=round(len(paraphrase_tokens) / len(source_tokens), 6),
        char_edit_distance=char_distance,
        normalized_char_edit_distance=round(
            char_distance / max(len(source), len(paraphrase)),
            6,
        ),
        token_edit_distance=token_distance,
        normalized_token_edit_distance=round(
            token_distance / max(len(source_tokens), len(paraphrase_tokens)),
            6,
        ),
        token_jaccard=round(len(token_intersection) / len(token_union), 6),
    )


def write_paraphrase_reports(
    input_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    records = list(read_models(input_path, PositiveParaphraseRecord))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "positive_paraphrase_metrics.csv"
    summary_path = output / "positive_paraphrase_summary.json"

    rows: list[dict[str, object]] = []
    for record in records:
        for item in record.paraphrases:
            rows.append(
                {
                    "sample_id": record.sample_id,
                    "pair_id": record.pair_id,
                    "positive_text": record.positive_text,
                    "paraphrase_id": item.paraphrase_id,
                    "paraphrase_text": item.text,
                    "rewrite_strategy": item.rewrite_strategy,
                    **item.metrics.model_dump(mode="json"),
                }
            )
    fieldnames = [
        "sample_id",
        "pair_id",
        "positive_text",
        "paraphrase_id",
        "paraphrase_text",
        "rewrite_strategy",
        "source_char_count",
        "paraphrase_char_count",
        "source_token_count",
        "paraphrase_token_count",
        "length_ratio",
        "char_edit_distance",
        "normalized_char_edit_distance",
        "token_edit_distance",
        "normalized_token_edit_distance",
        "token_jaccard",
    ]
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    generated = sum(record.metadata.num_generated for record in records)
    accepted = sum(record.metadata.num_accepted for record in records)
    failed = sum(
        record.metadata.operational_status == "failed" for record in records
    )
    summary = {
        "num_samples": len(records),
        "operationally_failed_samples": failed,
        "generated_candidates": generated,
        "accepted_paraphrases": accepted,
        "acceptance_rate": accepted / generated if generated else None,
        "samples_with_accepted_paraphrases": sum(
            bool(record.paraphrases) for record in records
        ),
        "mean_accepted_per_sample": accepted / len(records) if records else None,
        "mean_normalized_char_edit_distance": (
            mean(
                item.metrics.normalized_char_edit_distance
                for record in records
                for item in record.paraphrases
            )
            if accepted
            else None
        ),
        "mean_token_jaccard": (
            mean(
                item.metrics.token_jaccard
                for record in records
                for item in record.paraphrases
            )
            if accepted
            else None
        ),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"metrics": metrics_path, "summary": summary_path}


def _review_partition_error(
    response: ParaphraseReviewResponse,
    candidates: list[PositiveParaphrase],
) -> str | None:
    expected = {item.paraphrase_id for item in candidates}
    ids = [item.paraphrase_id for item in response.reviews]
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    unknown = sorted(set(ids) - expected)
    missing = sorted(expected - set(ids))
    problems: list[str] = []
    if duplicates:
        problems.append("duplicate ids: " + ", ".join(duplicates))
    if unknown:
        problems.append("unknown ids: " + ", ".join(unknown))
    if missing:
        problems.append("missing ids: " + ", ".join(missing))
    return "; ".join(problems) or None


def _build_fingerprint(config: PipelineConfig, requested: int) -> str:
    prompt_dir = Path(config.runtime.prompt_dir)
    version = config.runtime.prompt_version
    payload = {
        "schema": PARAPHRASE_SCHEMA_VERSION,
        "requested": requested,
        "generation_model": config.model.generation_model,
        "reviewer_model": config.model.validation_model,
        "structured_output_mode": config.model.structured_output_mode,
        "generation_reasoning_effort": config.model.generation_reasoning_effort,
        "review_reasoning_effort": config.model.validation_reasoning_effort,
        "paraphrase": config.paraphrase.model_dump(mode="json"),
        "generation_prompt": (
            prompt_dir / f"generate_positive_paraphrases_{version}.txt"
        ).read_text(encoding="utf-8"),
        "review_prompt": (
            prompt_dir / f"review_positive_paraphrases_{version}.txt"
        ).read_text(encoding="utf-8"),
    }
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _retry_predicate(
    *,
    prompt_version: str,
    build_fingerprint: str,
    requested: int,
    generator_model: str,
    reviewer_model: str,
):
    def should_retry(row: dict) -> bool:
        metadata = row.get("metadata", {})
        return bool(
            row.get("schema_version") != PARAPHRASE_SCHEMA_VERSION
            or metadata.get("prompt_version") != prompt_version
            or metadata.get("build_fingerprint") != build_fingerprint
            or metadata.get("requested_variants") != requested
            or metadata.get("generator_model") != generator_model
            or metadata.get("reviewer_model") != reviewer_model
            or metadata.get("operational_status") == "failed"
        )

    return should_retry


def _ordered_parallel(
    items: Iterable[T],
    function,
    max_workers: int,
) -> Iterator[T]:
    if max_workers <= 1:
        yield from map(function, items)
        return
    with ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="cirr-paraphrase",
    ) as executor:
        yield from executor.map(function, items)


def _clean_text(text: str) -> str:
    return " ".join(text.split())


def _normalize(text: str) -> str:
    return _clean_text(text).casefold()


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.casefold())


def _edit_distance(source: Sequence, target: Sequence) -> int:
    previous = list(range(len(target) + 1))
    for source_index, source_item in enumerate(source, start=1):
        current = [source_index]
        for target_index, target_item in enumerate(target, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[target_index] + 1,
                    previous[target_index - 1]
                    + (source_item != target_item),
                )
            )
        previous = current
    return previous[-1]
