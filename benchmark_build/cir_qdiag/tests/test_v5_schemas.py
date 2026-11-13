import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from cir_qdiag.schemas import (
    BatchReviewResponse,
    CIRRScreenResponse,
    GenerationBatchResponse,
    ImageRef,
    NegativeCandidate,
    SampleRecord,
)


def _candidate(status: str = "generated") -> NegativeCandidate:
    return NegativeCandidate(
        negative_id="x_object_add_text",
        text="Show a red car with a dog",
        semantic_target="object",
        edit_operation="add",
        construction_mode="text_only_editable",
        before_span=None,
        after_span="a dog",
        change_summary="add a dog",
        status=status,
        selection_reason="A valid decision." if status != "generated" else None,
    )


def test_generation_has_three_required_nullable_slots() -> None:
    schema = GenerationBatchResponse.model_json_schema()
    assert set(schema["required"]) == {"object", "attribute", "predicate"}
    proposal = schema["$defs"]["CandidateProposal"]
    assert set(proposal["required"]) == {
        "negative_text",
        "before_span",
        "after_span",
        "change_summary",
        "visual_basis",
    }


def test_edit_operation_requires_matching_anchor_shape() -> None:
    payload = _candidate().model_dump()
    payload.update(edit_operation="substitute", before_span=None, after_span="a dog")
    with pytest.raises(ValidationError, match="SUBSTITUTE requires"):
        NegativeCandidate.model_validate(payload)

    payload = _candidate().model_dump()
    payload.update(edit_operation="delete", before_span="red", after_span="blue")
    with pytest.raises(ValidationError, match="DELETE requires"):
        NegativeCandidate.model_validate(payload)


def test_batch_review_has_only_one_partition() -> None:
    required = set(BatchReviewResponse.model_json_schema()["required"])
    assert required == {"selected", "rejected"}


def test_source_screen_has_conservative_binary_decision() -> None:
    required = set(CIRRScreenResponse.model_json_schema()["required"])
    assert required == {"decision", "reason"}


def test_final_negatives_must_exactly_match_selected_candidates() -> None:
    accepted = _candidate("accepted")
    payload = {
        "sample_id": "s",
        "pair_id": 1,
        "reference": ImageRef(image_id="r", image_path="r.png").model_dump(),
        "target": ImageRef(image_id="t", image_path="t.png").model_dump(),
        "positive_text": "Show a red car",
        "candidates": [accepted.model_dump(mode="json")],
        "negatives": [],
        "metadata": {
            "stage": "final",
            "operational_status": "ok",
            "semantic_status": "accepted",
            "num_generated_candidates": 1,
            "num_accepted_negatives": 0,
        },
    }
    with pytest.raises(ValidationError, match="all and only accepted"):
        SampleRecord.model_validate(payload)


def test_only_v5_prompts_and_schemas_are_active() -> None:
    root = Path(__file__).parents[1]
    assert {path.name for path in (root / "prompts").glob("*.txt")} == {
        "screen_cirr_v5.txt",
        "generate_text_add_v5.txt",
        "generate_text_delete_v5.txt",
        "generate_text_substitute_v5.txt",
        "generate_vision_add_v5.txt",
        "generate_vision_delete_v5.txt",
        "generate_vision_substitute_v5.txt",
        "select_add_substitute_v5.txt",
        "select_delete_v5.txt",
        "generate_positive_paraphrases_v5.txt",
        "review_positive_paraphrases_v5.txt",
    }
    expected_schemas = {
        "cirr_screen_response.schema.json",
        "generation_batch_response.schema.json",
        "batch_review_response.schema.json",
        "negative_candidate.schema.json",
        "final_sample.schema.json",
        "paraphrase_generation_response.schema.json",
        "paraphrase_review_response.schema.json",
        "positive_paraphrase_record.schema.json",
    }
    assert {path.name for path in (root / "schemas").glob("*.json")} == expected_schemas
    final_schema = json.loads((root / "schemas" / "final_sample.schema.json").read_text())
    assert final_schema["properties"]["schema_version"]["const"] == "5.0"
    joined = json.dumps(final_schema)
    for removed in ("MutationSpec", "VisualContextAudit", "TextJudgment", "Claim"):
        assert removed not in joined
