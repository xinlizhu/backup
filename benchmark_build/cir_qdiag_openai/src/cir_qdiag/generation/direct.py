from __future__ import annotations

from pathlib import Path

from cir_qdiag.models.client import StructuredVLMClient
from cir_qdiag.schemas import (
    CallTrace,
    EditOperation,
    GenerationBatchResponse,
    InstructionRecord,
    NegativeCandidate,
    SemanticTarget,
)
from cir_qdiag.utils.prompts import dump_prompt_json, load_prompt


def generate_operation(
    group_id: str,
    instruction: InstructionRecord,
    operation: EditOperation,
    client: StructuredVLMClient,
    prompt_dir: str | Path,
    prompt_version: str,
    max_output_tokens: int,
) -> tuple[list[NegativeCandidate], CallTrace]:
    """Generate object/attribute/predicate edits from text only."""

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
                "instruction_id": instruction.instruction_id,
                "source_instruction": instruction.text,
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
        before_span = _optional_text(proposal.before_span)
        after_span = _optional_text(proposal.after_span)
        candidates.append(
            NegativeCandidate(
                negative_id=(
                    f"{group_id}_{instruction.instruction_id}_"
                    f"{target.value}_{operation.value}"
                ),
                instruction_id=instruction.instruction_id,
                source_text=instruction.text,
                text=text,
                semantic_target=target,
                edit_operation=operation,
                before_span=before_span,
                after_span=after_span,
                generation_attempts=trace.attempts,
            )
        )
    return candidates, trace


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = " ".join(value.split())
    return stripped or None
