from __future__ import annotations

from pathlib import Path

from cir_qdiag.models.client import StructuredVLMClient
from cir_qdiag.schemas import (
    CallTrace,
    GroupReviewResponse,
    ImageRef,
    NegativeCandidate,
    PinPointGroupRecord,
)
from cir_qdiag.utils.hashing import sha256_text
from cir_qdiag.utils.prompts import dump_prompt_json, load_prompt


def review_group_candidates(
    record: PinPointGroupRecord,
    client: StructuredVLMClient,
    prompt_dir: str | Path,
    prompt_version: str,
    max_output_tokens: int,
    positive_images: list[ImageRef],
) -> tuple[list[NegativeCandidate], list[CallTrace]]:
    """Validate every generated text for one group in exactly one multimodal call."""

    if not record.candidates:
        return [], []
    positive_ids = {item.image_id for item in record.positive_images}
    if not positive_images or any(
        item.image_id not in positive_ids for item in positive_images
    ):
        raise ValueError("validation positives must be a non-empty subset of the group")
    prompt, _ = load_prompt(prompt_dir, "validate_group", prompt_version)
    reference_count = len(record.references)
    positive_count = len(positive_images)
    payload = {
        "group_id": record.group_id,
        "image_layout": {
            "reference_images": list(range(1, reference_count + 1)),
            "positive_images": list(
                range(reference_count + 1, reference_count + positive_count + 1)
            ),
        },
        "instructions": [
            {
                "instruction_id": item.instruction_id,
                "text": item.text,
            }
            for item in record.instructions
        ],
        "candidates": [
            {
                "negative_id": item.negative_id,
                "instruction_id": item.instruction_id,
                "source_text": item.source_text,
                "negative_text": item.text,
                "semantic_target": item.semantic_target.value,
                "edit_operation": item.edit_operation.value,
                "before_span": item.before_span,
                "after_span": item.after_span,
            }
            for item in record.candidates
        ],
    }
    images = [
        item.image_path for item in [*record.references, *positive_images]
    ]
    traces: list[CallTrace] = []
    errors: list[str] = []
    for _ in range(client.max_retries + 1):
        response, trace = client.call(
            GroupReviewResponse,
            images=images,
            system_prompt=prompt,
            user_prompt=dump_prompt_json(payload),
            stage="validate_group",
            max_output_tokens=max_output_tokens,
        )
        traces.append(trace)
        error = _verdict_error(response, record.candidates)
        if error is None:
            return _apply_reviews(response, record.candidates), traces
        errors.append(error)
        client.reject_cached_response(trace, error)
    raise RuntimeError(
        "validate_group returned an invalid verdict partition: " + "; ".join(errors)
    )


def select_validation_positives(
    record: PinPointGroupRecord,
    max_positive_images: int,
) -> list[ImageRef]:
    """Choose a stable, filename-unbiased subset for group validation."""

    if max_positive_images < 1:
        raise ValueError("max_positive_images must be positive")
    if len(record.positive_images) <= max_positive_images:
        return list(record.positive_images)
    return sorted(
        record.positive_images,
        key=lambda item: sha256_text(f"{record.group_id}\0{item.image_id}"),
    )[:max_positive_images]


def _verdict_error(
    response: GroupReviewResponse,
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
    response: GroupReviewResponse,
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
