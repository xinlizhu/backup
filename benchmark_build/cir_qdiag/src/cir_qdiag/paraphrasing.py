from __future__ import annotations

import csv
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean, median
from typing import Callable, Iterable, Iterator, Sequence, TypeVar

from tqdm.auto import tqdm

from cir_qdiag.config import PipelineConfig
from cir_qdiag.models.client import StructuredVLMClient
from cir_qdiag.schemas import (
    PARAPHRASE_SCHEMA_VERSION,
    CallTrace,
    ParaphraseGenerationResponse,
    ParaphraseMetrics,
    ParaphraseReviewResponse,
    PositiveParaphrase,
    PositiveParaphraseMetadata,
    PositiveParaphraseRecord,
    SampleRecord,
)
from cir_qdiag.utils.hashing import deterministic_seed, sha256_text
from cir_qdiag.utils.jsonl import ResumableJsonlWriter, read_models
from cir_qdiag.utils.prompts import dump_prompt_json, load_prompt


LOGGER = logging.getLogger(__name__)
T = TypeVar("T")


def positive_paraphrase_stage(
    config: PipelineConfig,
    input_path: str | Path,
    output_path: str | Path,
    client: StructuredVLMClient | None = None,
    *,
    force: bool = False,
    num_variants: int | None = None,
) -> Path:
    """Generate and textually verify positive paraphrases without running negative stages."""

    if client is None:
        # Imported lazily to keep this module independent from the main pipeline at import time.
        from cir_qdiag.pipeline import create_client

        client = create_client(config)
    requested = num_variants or config.paraphrase.num_variants
    if not 1 <= requested <= 20:
        raise ValueError("num_variants must be between 1 and 20")

    records = list(read_models(input_path, SampleRecord))
    output = Path(output_path)
    fingerprint = _paraphrase_build_fingerprint(config, requested)
    with ResumableJsonlWriter(
        output,
        force=force,
        order_ids=[record.sample_id for record in records],
        retry_predicate=_retry_predicate(
            prompt_version=config.runtime.prompt_version,
            build_fingerprint=fingerprint,
            requested=requested,
            model_name=config.model.name,
        ),
    ) as writer:
        pending = [record for record in records if not writer.has(record.sample_id)]

        def process(record: SampleRecord) -> PositiveParaphraseRecord:
            metadata = PositiveParaphraseMetadata(
                requested_variants=requested,
                model_name=config.model.name,
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
                    temperature=config.paraphrase.generation_temperature,
                    seed=deterministic_seed(
                        config.sampling.seed,
                        record.sample_id,
                        "generate_positive_paraphrases",
                    ),
                    max_new_tokens=config.paraphrase.generation_max_new_tokens,
                )
                result.metadata.call_traces.append(generation_trace)
                if len(candidates) < requested:
                    result.metadata.notes.append(
                        f"generated {len(candidates)} unique rewrites out of {requested} requested"
                    )
                result.candidates = candidates
                result.metadata.num_generated = len(candidates)
                if not candidates:
                    raise RuntimeError("Qwen returned no non-identical unique paraphrases")

                reviewed, review_traces = review_positive_paraphrases(
                    positive_text=record.positive_text,
                    candidates=candidates,
                    client=client,
                    prompt_dir=config.runtime.prompt_dir,
                    prompt_version=config.runtime.prompt_version,
                    temperature=config.paraphrase.review_temperature,
                    seed=deterministic_seed(
                        config.sampling.seed,
                        record.sample_id,
                        "review_positive_paraphrases",
                    ),
                    max_new_tokens=config.paraphrase.review_max_new_tokens,
                )
                result.metadata.call_traces.extend(review_traces)
                result.candidates = reviewed
                result.paraphrases = [
                    candidate for candidate in reviewed if candidate.status == "accepted"
                ]
                result.metadata.num_accepted = len(result.paraphrases)
            except Exception as exc:
                result.metadata.operational_status = "failed"
                result.metadata.errors.append(f"{type(exc).__name__}:{exc}")
                LOGGER.exception("positive paraphrasing failed for %s", record.sample_id)
            return PositiveParaphraseRecord.model_validate(result.model_dump(mode="json"))

        completed = len(records) - len(pending)
        results = _ordered_parallel(
            pending,
            process,
            1 if config.model.backend == "transformers" else config.runtime.max_workers,
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
                    refresh=False,
                )
    LOGGER.info("wrote positive paraphrases for %d samples to %s", len(records), output)
    return output


def generate_positive_paraphrases(
    *,
    record: SampleRecord,
    client: StructuredVLMClient,
    prompt_dir: str | Path,
    prompt_version: str,
    requested: int,
    temperature: float,
    seed: int,
    max_new_tokens: int,
) -> tuple[list[PositiveParaphrase], CallTrace]:
    prompt, _ = load_prompt(prompt_dir, "generate_positive_paraphrases", prompt_version)
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
        temperature=temperature,
        seed=seed,
        max_new_tokens=max_new_tokens,
    )

    source_normalized = _normalize(record.positive_text)
    seen = {source_normalized}
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
    client: StructuredVLMClient,
    prompt_dir: str | Path,
    prompt_version: str,
    temperature: float,
    seed: int,
    max_new_tokens: int,
) -> tuple[list[PositiveParaphrase], list[CallTrace]]:
    prompt, _ = load_prompt(prompt_dir, "review_positive_paraphrases", prompt_version)
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
    for attempt in range(client.max_retries + 1):
        response, trace = client.call(
            ParaphraseReviewResponse,
            images=[],
            system_prompt=prompt,
            user_prompt=dump_prompt_json(payload),
            stage="review_positive_paraphrases",
            temperature=temperature,
            seed=seed + attempt,
            max_new_tokens=max_new_tokens,
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
                    "accepted" if decision.decision == "equivalent" else "rejected"
                )
                result.review_reason = decision.reason
                reviewed.append(result)
            return reviewed, traces
        errors.append(error)
        client.reject_cached_response(trace, error)
    raise RuntimeError(
        "review_positive_paraphrases returned an invalid partition: " + "; ".join(errors)
    )


def compute_paraphrase_metrics(source_text: str, paraphrase_text: str) -> ParaphraseMetrics:
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
        token_jaccard=round(
            len(token_intersection) / len(token_union),
            6,
        ),
    )


def write_paraphrase_reports(
    input_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    records = list(read_models(input_path, PositiveParaphraseRecord))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "positive_paraphrase_metrics.csv"
    summary_path = output_dir / "positive_paraphrase_summary.json"

    fieldnames = [
        "sample_id",
        "pair_id",
        "positive_text",
        "paraphrase_id",
        "paraphrase_text",
        "rewrite_strategy",
        "length_ratio",
        "char_edit_distance",
        "normalized_char_edit_distance",
        "token_edit_distance",
        "normalized_token_edit_distance",
        "token_jaccard",
    ]
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
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    metric_names = [
        "length_ratio",
        "normalized_char_edit_distance",
        "normalized_token_edit_distance",
        "token_jaccard",
    ]
    summary = {
        "num_samples": len(records),
        "num_failed_samples": sum(
            record.metadata.operational_status == "failed" for record in records
        ),
        "num_generated": sum(record.metadata.num_generated for record in records),
        "num_accepted": sum(record.metadata.num_accepted for record in records),
        "acceptance_rate": round(
            sum(record.metadata.num_accepted for record in records)
            / max(1, sum(record.metadata.num_generated for record in records)),
            6,
        ),
        "accepted_perturbation_metrics": {
            name: _summarize([float(row[name]) for row in rows]) for name in metric_names
        },
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
    expected = {candidate.paraphrase_id for candidate in candidates}
    ids = [review.paraphrase_id for review in response.reviews]
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


def _paraphrase_build_fingerprint(config: PipelineConfig, requested: int) -> str:
    generation_prompt, _ = load_prompt(
        config.runtime.prompt_dir,
        "generate_positive_paraphrases",
        config.runtime.prompt_version,
    )
    review_prompt, _ = load_prompt(
        config.runtime.prompt_dir,
        "review_positive_paraphrases",
        config.runtime.prompt_version,
    )
    payload = {
        "schema_version": PARAPHRASE_SCHEMA_VERSION,
        "prompt_version": config.runtime.prompt_version,
        "requested": requested,
        "model": config.model.model_dump(mode="json"),
        "paraphrase": config.paraphrase.model_dump(mode="json"),
        "generation_prompt": generation_prompt,
        "review_prompt": review_prompt,
    }
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))


def _retry_predicate(
    *,
    prompt_version: str,
    build_fingerprint: str,
    requested: int,
    model_name: str,
) -> Callable[[dict], bool]:
    def should_retry(row: dict) -> bool:
        metadata = row.get("metadata", {})
        return bool(
            row.get("schema_version") != PARAPHRASE_SCHEMA_VERSION
            or metadata.get("prompt_version") != prompt_version
            or metadata.get("build_fingerprint") != build_fingerprint
            or metadata.get("requested_variants") != requested
            or metadata.get("model_name") != model_name
            or metadata.get("operational_status") == "failed"
        )

    return should_retry


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
        thread_name_prefix="cir-qdiag-paraphrase",
    ) as executor:
        yield from executor.map(function, items)


def _summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p10": None,
            "p90": None,
            "min": None,
            "max": None,
        }
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": round(mean(ordered), 6),
        "median": round(median(ordered), 6),
        "p10": round(_quantile(ordered, 0.1), 6),
        "p90": round(_quantile(ordered, 0.9), 6),
        "min": round(ordered[0], 6),
        "max": round(ordered[-1], 6),
    }


def _quantile(ordered: list[float], probability: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _clean_text(text: str) -> str:
    return " ".join(text.split())


def _normalize(text: str) -> str:
    return _clean_text(text).casefold()


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.casefold(), flags=re.UNICODE)


def _edit_distance(source: Sequence[str], target: Sequence[str]) -> int:
    if len(source) < len(target):
        source, target = target, source
    previous = list(range(len(target) + 1))
    for source_index, source_item in enumerate(source, start=1):
        current = [source_index]
        for target_index, target_item in enumerate(target, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[target_index] + 1,
                    previous[target_index - 1] + (source_item != target_item),
                )
            )
        previous = current
    return previous[-1]
