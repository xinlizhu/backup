from __future__ import annotations

import re
from pathlib import Path

from cir_qdiag.models.client import StructuredVLMClient
from cir_qdiag.schemas import (
    CallTrace,
    ConstructionMode,
    EditOperation,
    GenerationBatchResponse,
    NegativeCandidate,
    SampleRecord,
    SemanticTarget,
)
from cir_qdiag.utils.prompts import dump_prompt_json, load_prompt


def generate_operation(
    record: SampleRecord,
    mode: ConstructionMode,
    operation: EditOperation,
    client: StructuredVLMClient,
    prompt_dir: str | Path,
    prompt_version: str,
    temperature: float,
    seed: int,
    max_new_tokens: int,
) -> tuple[list[NegativeCandidate], CallTrace]:
    """Directly generate up to three captions for one fixed mode and operation."""

    route = "text" if mode == ConstructionMode.TEXT_ONLY else "vision"
    prompt, _ = load_prompt(
        prompt_dir,
        f"generate_{route}_{operation.value}",
        prompt_version,
    )
    response, trace = client.call(
        GenerationBatchResponse,
        images=(
            []
            if mode == ConstructionMode.TEXT_ONLY
            else [record.target.image_path]
        ),
        system_prompt=prompt,
        user_prompt=dump_prompt_json(
            {
                "positive_text": record.positive_text,
                "construction_mode": mode.value,
                "edit_operation": operation.value,
            }
        ),
        stage=f"generate_{route}_{operation.value}",
        temperature=temperature,
        seed=seed,
        max_new_tokens=max_new_tokens,
    )

    candidates: list[NegativeCandidate] = []
    positive = _normalize(record.positive_text)
    for target in SemanticTarget:
        proposal = getattr(response, target.value)
        if proposal is None:
            continue
        text = " ".join(proposal.negative_text.split())
        summary = " ".join(proposal.change_summary.split())
        if not text or not summary or _normalize(text) == positive:
            continue
        before_span = _optional_text(proposal.before_span)
        after_span = _optional_text(proposal.after_span)
        if not _anchors_match(
            record.positive_text,
            text,
            operation,
            before_span,
            after_span,
        ):
            continue
        visual_basis = _optional_text(proposal.visual_basis)
        if mode == ConstructionMode.TEXT_ONLY:
            visual_basis = None
        mode_suffix = "text" if mode == ConstructionMode.TEXT_ONLY else "vision"
        candidates.append(
            NegativeCandidate(
                negative_id=(
                    f"{record.sample_id}_{target.value}_{operation.value}_{mode_suffix}"
                ),
                text=text,
                semantic_target=target,
                edit_operation=operation,
                construction_mode=mode,
                before_span=before_span,
                after_span=after_span,
                change_summary=summary,
                visual_basis=visual_basis,
                generation_attempts=trace.attempts,
            )
        )
    return candidates, trace


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = " ".join(value.split())
    return stripped or None


def _anchors_match(
    positive_text: str,
    negative_text: str,
    operation: EditOperation,
    before_span: str | None,
    after_span: str | None,
) -> bool:
    if operation == EditOperation.ADD:
        return bool(
            before_span is None
            and after_span
            and _contains_span(negative_text, after_span)
            and not _contains_span(positive_text, after_span)
        )
    if operation == EditOperation.DELETE:
        return bool(
            before_span
            and after_span is None
            and _contains_span(positive_text, before_span)
            and not _contains_span(negative_text, before_span)
        )
    return bool(
        before_span
        and after_span
        and _normalize(before_span) != _normalize(after_span)
        and _contains_span(positive_text, before_span)
        and _contains_span(negative_text, after_span)
    )


def _contains_span(text: str, span: str) -> bool:
    normalized_text = _normalize(text)
    normalized_span = _normalize(span)
    if not normalized_span:
        return False
    return re.search(
        rf"(?<!\w){re.escape(normalized_span)}(?!\w)",
        normalized_text,
    ) is not None
