from __future__ import annotations

import pytest
from pydantic import ValidationError

from cir_qdiag.schemas import (
    GroupReviewResponse,
    ImageRef,
    InstructionRecord,
    NegativeCandidate,
    PinPointGroupRecord,
)


def _instruction(i: int) -> InstructionRecord:
    return InstructionRecord(
        instruction_id=f"q{i}",
        text="Show a red dog",
        query_category="Swap",
        token_count=4,
        length_category="short",
        is_question=False,
        is_imperative=True,
        l1_interest="Animals",
    )


def test_group_requires_two_to_six_instructions() -> None:
    common = {
        "group_id": "g",
        "references": [ImageRef(image_id="r", image_path="r.jpg")],
        "positive_images": [ImageRef(image_id="p", image_path="p.jpg")],
        "official_negative_image_ids": ["n"],
    }
    with pytest.raises(ValidationError):
        PinPointGroupRecord(**common, instructions=[_instruction(1)])
    PinPointGroupRecord(**common, instructions=[_instruction(1), _instruction(2)])


def test_candidate_keeps_untrusted_spans_for_model_review() -> None:
    candidate = NegativeCandidate(
        negative_id="g_q1_object_add",
        instruction_id="q1",
        source_text="Show a red dog",
        text="Show a red dog with a ball",
        semantic_target="object",
        edit_operation="add",
        before_span=None,
        after_span="with a ball",
    )
    assert candidate.status == "generated"
    malformed = NegativeCandidate.model_validate(
        candidate.model_dump()
        | {
            "text": candidate.source_text,
            "before_span": None,
            "after_span": None,
        }
    )
    assert malformed.text == malformed.source_text
    assert malformed.after_span is None


def test_group_review_requires_failure_code_only_on_rejection() -> None:
    GroupReviewResponse.model_validate(
        {
            "verdicts": [
                {
                    "negative_id": "n",
                    "decision": "accept",
                    "failure_code": None,
                    "reason": "Valid.",
                }
            ]
        }
    )
    with pytest.raises(ValidationError):
        GroupReviewResponse.model_validate(
            {
                "verdicts": [
                    {
                        "negative_id": "n",
                        "decision": "reject",
                        "failure_code": None,
                        "reason": "Invalid.",
                    }
                ]
            }
        )


def test_validated_positive_ids_must_be_a_subset_and_at_most_eight() -> None:
    common = {
        "group_id": "g",
        "references": [ImageRef(image_id="r", image_path="r.jpg")],
        "instructions": [_instruction(1), _instruction(2)],
        "positive_images": [
            ImageRef(image_id=f"p{i}", image_path=f"p{i}.jpg")
            for i in range(9)
        ],
        "official_negative_image_ids": ["n"],
    }
    PinPointGroupRecord.model_validate(
        common
        | {
            "metadata": {
                "validated_positive_image_ids": [f"p{i}" for i in range(8)]
            }
        }
    )
    with pytest.raises(ValidationError):
        PinPointGroupRecord.model_validate(
            common
            | {
                "metadata": {
                    "validated_positive_image_ids": [f"p{i}" for i in range(9)]
                }
            }
        )
    with pytest.raises(ValidationError, match="must belong"):
        PinPointGroupRecord.model_validate(
            common | {"metadata": {"validated_positive_image_ids": ["unknown"]}}
        )
