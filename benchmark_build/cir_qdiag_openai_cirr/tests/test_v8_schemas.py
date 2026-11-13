from __future__ import annotations

import pytest
from pydantic import ValidationError

from cir_qdiag_openai_cirr.schemas import (
    CIRRSampleRecord,
    ImageRef,
    NegativeCandidate,
    PositiveParaphraseMetadata,
    PositiveParaphraseRecord,
    SampleReviewResponse,
)


def _sample(**updates) -> CIRRSampleRecord:
    payload = {
        "sample_id": "cirr_train_7",
        "pair_id": 7,
        "reference": ImageRef(image_id="r", image_path="r.png"),
        "target": ImageRef(image_id="t", image_path="t.png"),
        "positive_text": "Make the red dog run",
        "image_set": ["r", "t"],
    }
    payload.update(updates)
    return CIRRSampleRecord.model_validate(payload)


def test_sample_identity_is_cirr_train_pair_id() -> None:
    assert _sample().schema_version == "8.0"
    with pytest.raises(ValidationError, match="sample_id"):
        _sample(sample_id="cirr_train_8")


def test_candidate_is_training_manifest_compatible() -> None:
    candidate = NegativeCandidate(
        negative_id="cirr_train_7_object_add",
        source_text="Make the red dog run",
        text="Make the red dog run with a ball",
        semantic_target="object",
        edit_operation="add",
        before_span=None,
        after_span="with a ball",
    )
    assert candidate.construction_mode == "text_only_editable"
    assert candidate.status == "generated"


def test_candidate_axis_must_be_unique_within_sample() -> None:
    candidate = NegativeCandidate(
        negative_id="n1",
        source_text="Make the red dog run",
        text="Make the blue dog run",
        semantic_target="attribute",
        edit_operation="substitute",
        before_span="red",
        after_span="blue",
    )
    duplicate = candidate.model_copy(update={"negative_id": "n2"})
    with pytest.raises(ValidationError, match="taxonomy cell"):
        _sample(candidates=[candidate, duplicate])


def test_text_review_requires_failure_code_only_on_rejection() -> None:
    SampleReviewResponse.model_validate(
        {
            "verdicts": [
                {
                    "negative_id": "n",
                    "decision": "accept",
                    "failure_code": None,
                    "reason": "Valid single semantic edit.",
                }
            ]
        }
    )
    with pytest.raises(ValidationError):
        SampleReviewResponse.model_validate(
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


def test_positive_paraphrase_record_contains_only_accepted_items() -> None:
    metrics = {
        "source_char_count": 8,
        "paraphrase_char_count": 8,
        "source_token_count": 2,
        "paraphrase_token_count": 2,
        "length_ratio": 1.0,
        "char_edit_distance": 2,
        "normalized_char_edit_distance": 0.25,
        "token_edit_distance": 1,
        "normalized_token_edit_distance": 0.5,
        "token_jaccard": 0.5,
    }
    accepted = {
        "paraphrase_id": "cirr_train_7_p1",
        "text": "Have it run",
        "rewrite_strategy": "Verb rewording",
        "metrics": metrics,
        "status": "accepted",
        "review_reason": "Equivalent.",
    }
    record = PositiveParaphraseRecord(
        sample_id="cirr_train_7",
        pair_id=7,
        reference={"image_id": "r", "image_path": "r.png"},
        target={"image_id": "t", "image_path": "t.png"},
        positive_text="Make it run",
        candidates=[accepted],
        paraphrases=[accepted],
        metadata=PositiveParaphraseMetadata(
            requested_variants=1,
            num_generated=1,
            num_accepted=1,
            generator_model="gpt",
            reviewer_model="gpt",
            prompt_version="v8",
            build_fingerprint="fingerprint",
        ),
    )
    assert record.paraphrases[0].status == "accepted"
