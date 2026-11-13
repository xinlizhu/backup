from __future__ import annotations

from pathlib import Path

from cir_qdiag_openai_cirr.models.client import StructuredLLMClient
from cir_qdiag_openai_cirr.schemas import (
    CallTrace,
    CIRRSampleRecord,
    NegativeCandidate,
    SampleReviewResponse,
)
from cir_qdiag_openai_cirr.utils.prompts import dump_prompt_json, load_prompt


def review_sample_candidates(
    record: CIRRSampleRecord,
    client: StructuredLLMClient,
    prompt_dir: str | Path,
    prompt_version: str,
    max_output_tokens: int,
) -> tuple[list[NegativeCandidate], list[CallTrace]]:
    """Review one sample using its Reference and Target images."""

    if not record.candidates:
        return [], []
    prompt, _ = load_prompt(prompt_dir, "validate_visual", prompt_version)
    payload = {
        "sample_id": record.sample_id,
        "image_layout": {
            "reference_image": 1,
            "target_positive_image": 2,
        },
        "source_instruction": record.positive_text,
        "candidates": [
            {
                "negative_id": item.negative_id,
                "negative_text": item.text,
                "semantic_target": item.semantic_target.value,
                "edit_operation": item.edit_operation.value,
                "before_span": item.before_span,
                "after_span": item.after_span,
            }
            for item in record.candidates
        ],
    }
    traces: list[CallTrace] = []
    errors: list[str] = []
    for _ in range(client.max_retries + 1):
        response, trace = client.call(
            SampleReviewResponse,
            images=[record.reference.image_path, record.target.image_path],
            system_prompt=prompt,
            user_prompt=dump_prompt_json(payload),
            stage="validate_visual",
            max_output_tokens=max_output_tokens,
        )
        traces.append(trace)
        error = _verdict_error(response, record.candidates)
        if error is None:
            return _apply_reviews(response, record.candidates), traces
        errors.append(error)
        client.reject_cached_response(trace, error)
    raise RuntimeError(
        "validate_visual returned an invalid verdict partition: " + "; ".join(errors)
    )


def _verdict_error(
    response: SampleReviewResponse,
    candidates: list[NegativeCandidate],
) -> str | None:
    expected = {item.negative_id for item in candidates}
    ids = [item.negative_id for item in response.verdicts]
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
    response: SampleReviewResponse,
    candidates: list[NegativeCandidate],
) -> list[NegativeCandidate]:
    verdicts = {item.negative_id: item for item in response.verdicts}
    reviewed: list[NegativeCandidate] = []
    for candidate in candidates:
        verdict = verdicts[candidate.negative_id]
        payload = candidate.model_dump(mode="json")
        if verdict.decision == "accept":
            payload.update(status="accepted", rejection_code=None)
        else:
            payload.update(status="rejected", rejection_code=verdict.failure_code)
        payload["selection_reason"] = verdict.reason
        reviewed.append(NegativeCandidate.model_validate(payload))
    return reviewed
