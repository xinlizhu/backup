from __future__ import annotations

from pathlib import Path

from cir_qdiag.models.client import StructuredVLMClient
from cir_qdiag.schemas import (
    BatchReviewResponse,
    CallTrace,
    EditOperation,
    NegativeCandidate,
    SampleRecord,
)
from cir_qdiag.utils.prompts import dump_prompt_json, load_prompt


def review_candidates(
    record: SampleRecord,
    client: StructuredVLMClient,
    prompt_dir: str | Path,
    prompt_version: str,
    temperature: float,
    seed: int,
    max_new_tokens: int,
) -> tuple[list[NegativeCandidate], list[CallTrace]]:
    """Review ADD/SUBSTITUTE and DELETE with separate decision contracts."""

    add_substitute = [
        item for item in record.candidates if item.edit_operation != EditOperation.DELETE
    ]
    deletes = [
        item for item in record.candidates if item.edit_operation == EditOperation.DELETE
    ]
    reviewed_by_id: dict[str, NegativeCandidate] = {}
    traces: list[CallTrace] = []

    for candidates, prompt_stem, stage, route_seed in (
        (
            add_substitute,
            "select_add_substitute",
            "select_add_substitute",
            seed,
        ),
        (deletes, "select_delete", "select_delete", seed + 10_000),
    ):
        if not candidates:
            continue
        reviewed, route_traces = _review_partition(
            record=record,
            candidates=candidates,
            client=client,
            prompt_dir=prompt_dir,
            prompt_version=prompt_version,
            prompt_stem=prompt_stem,
            stage=stage,
            temperature=temperature,
            seed=route_seed,
            max_new_tokens=max_new_tokens,
        )
        reviewed_by_id.update({item.negative_id: item for item in reviewed})
        traces.extend(route_traces)

    return [reviewed_by_id[item.negative_id] for item in record.candidates], traces


def _review_partition(
    *,
    record: SampleRecord,
    candidates: list[NegativeCandidate],
    client: StructuredVLMClient,
    prompt_dir: str | Path,
    prompt_version: str,
    prompt_stem: str,
    stage: str,
    temperature: float,
    seed: int,
    max_new_tokens: int,
) -> tuple[list[NegativeCandidate], list[CallTrace]]:
    prompt, _ = load_prompt(prompt_dir, prompt_stem, prompt_version)
    payload = {
        "positive_text": record.positive_text,
        "candidates": [
            {
                "negative_id": item.negative_id,
                "text": item.text,
                "semantic_target": item.semantic_target.value,
                "edit_operation": item.edit_operation.value,
                "construction_mode": item.construction_mode.value,
                "before_span": item.before_span,
                "after_span": item.after_span,
                "change_summary": item.change_summary,
                "visual_basis": item.visual_basis,
            }
            for item in candidates
        ],
    }
    traces: list[CallTrace] = []
    errors: list[str] = []
    for attempt in range(client.max_retries + 1):
        response, trace = client.call(
            BatchReviewResponse,
            images=[record.target.image_path],
            system_prompt=prompt,
            user_prompt=dump_prompt_json(payload),
            stage=stage,
            temperature=temperature,
            seed=seed + attempt,
            max_new_tokens=max_new_tokens,
        )
        traces.append(trace)
        error = _partition_error(response, candidates)
        if error is None:
            return _apply_reviews(response, candidates), traces
        errors.append(error)
        client.reject_cached_response(trace, error)
    raise RuntimeError(f"{stage} returned an invalid partition: " + "; ".join(errors))


def _partition_error(
    response: BatchReviewResponse,
    candidates: list[NegativeCandidate],
) -> str | None:
    expected = {item.negative_id for item in candidates}
    ids = [item.negative_id for item in response.selected + response.rejected]
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


def _apply_reviews(
    response: BatchReviewResponse,
    candidates: list[NegativeCandidate],
) -> list[NegativeCandidate]:
    selected = {item.negative_id: item.reason for item in response.selected}
    rejected = {item.negative_id: item.reason for item in response.rejected}
    reviewed: list[NegativeCandidate] = []
    for candidate in candidates:
        result = candidate.model_copy(deep=True)
        if candidate.negative_id in selected:
            result.status = "accepted"
            result.selection_reason = selected[candidate.negative_id]
        else:
            result.status = "rejected"
            result.selection_reason = rejected[candidate.negative_id]
        reviewed.append(result)
    return reviewed
