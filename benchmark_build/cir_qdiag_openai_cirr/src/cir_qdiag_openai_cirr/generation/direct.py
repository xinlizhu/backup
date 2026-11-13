from __future__ import annotations

from pathlib import Path

from cir_qdiag_openai_cirr.models.client import StructuredLLMClient
from cir_qdiag_openai_cirr.schemas import (
    CallTrace,
    CIRRSampleRecord,
    EditOperation,
    GenerationBatchResponse,
    NegativeCandidate,
    SemanticTarget,
)
from cir_qdiag_openai_cirr.utils.prompts import dump_prompt_json, load_prompt


def generate_operation(
    record: CIRRSampleRecord,
    operation: EditOperation,
    client: StructuredLLMClient,
    prompt_dir: str | Path,
    prompt_version: str,
    max_output_tokens: int,
) -> tuple[list[NegativeCandidate], CallTrace]:
    """Generate up to three counterfactual texts without sending CIRR images."""

    prompt, _ = load_prompt(
        prompt_dir,
        f"generate_{operation.value}",
        prompt_version,
    )
    response, trace = client.call(
        GenerationBatchResponse,
        images=[],
        system_prompt=prompt,
        user_prompt=dump_prompt_json(
            {
                "sample_id": record.sample_id,
                "source_instruction": record.positive_text,
                "edit_operation": operation.value,
            }
        ),
        stage=f"generate_{operation.value}",
        max_output_tokens=max_output_tokens,
    )

    candidates: list[NegativeCandidate] = []
    for target in SemanticTarget:
        proposal = getattr(response, target.value)
        if proposal is None:
            continue
        text = " ".join(proposal.negative_text.split())
        if not text:
            continue
        candidates.append(
            NegativeCandidate(
                negative_id=f"{record.sample_id}_{target.value}_{operation.value}",
                source_text=record.positive_text,
                text=text,
                semantic_target=target,
                edit_operation=operation,
                before_span=_optional_text(proposal.before_span),
                after_span=_optional_text(proposal.after_span),
                generation_attempts=trace.attempts,
            )
        )
    return candidates, trace


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = " ".join(value.split())
    return stripped or None
