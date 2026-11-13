import json
from pathlib import Path

from cir_qdiag.models.client import StructuredVLMClient
from cir_qdiag.models.fake import FakeVLMBackend
from cir_qdiag.schemas import ImageRef, NegativeCandidate, SampleRecord
from cir_qdiag.validation.batch_review import review_candidates


def _add_substitute_candidate() -> NegativeCandidate:
    return NegativeCandidate(
        negative_id="cirr_val_1_attribute_substitute_text",
        text="Show a blue car",
        semantic_target="attribute",
        edit_operation="substitute",
        construction_mode="text_only_editable",
        before_span="red",
        after_span="blue",
        change_summary="red to blue",
    )


def _delete_candidate() -> NegativeCandidate:
    return NegativeCandidate(
        negative_id="cirr_val_1_attribute_delete_text",
        text="Show a car",
        semantic_target="attribute",
        edit_operation="delete",
        construction_mode="text_only_editable",
        before_span="red",
        after_span=None,
        change_summary="remove red",
    )


def _record(candidates: list[NegativeCandidate] | None = None) -> SampleRecord:
    return SampleRecord(
        sample_id="cirr_val_1",
        pair_id=1,
        reference=ImageRef(image_id="r", image_path="ref.png"),
        target=ImageRef(image_id="t", image_path="target.png"),
        positive_text="Show a red car",
        candidates=candidates or [_add_substitute_candidate()],
    )


def test_add_substitute_candidates_use_their_own_review(tmp_path: Path) -> None:
    response = {
        "selected": [
            {
                "negative_id": "cirr_val_1_attribute_substitute_text",
                "reason": "The blue color contradicts the red target car.",
            }
        ],
        "rejected": [],
    }
    backend = FakeVLMBackend([response])
    client = StructuredVLMClient(backend, tmp_path, prompt_version="v5", max_retries=0)
    reviewed, traces = review_candidates(
        _record(),
        client,
        Path(__file__).parents[1] / "prompts",
        "v5",
        0.0,
        42,
        512,
    )
    assert len(traces) == 1
    assert len(backend.calls) == 1
    assert backend.calls[0]["stage"] == "select_add_substitute"
    assert backend.calls[0]["images"] == ["target.png"]
    assert "exactly one image: the TARGET image" in backend.calls[0]["system_prompt"]
    payload = json.loads(backend.calls[0]["user_prompt"])["candidates"][0]
    assert payload["before_span"] == "red"
    assert payload["after_span"] == "blue"
    assert reviewed[0].status == "accepted"


def test_delete_candidates_use_a_separate_review_contract(tmp_path: Path) -> None:
    add_substitute_response = {
        "selected": [
            {
                "negative_id": "cirr_val_1_attribute_substitute_text",
                "reason": "Blue contradicts the red target car.",
            }
        ],
        "rejected": [],
    }
    delete_response = {
        "selected": [
            {
                "negative_id": "cirr_val_1_attribute_delete_text",
                "reason": "It removes red and is strictly weaker.",
            }
        ],
        "rejected": [],
    }
    backend = FakeVLMBackend([add_substitute_response, delete_response])
    client = StructuredVLMClient(backend, tmp_path, prompt_version="v5", max_retries=0)
    reviewed, traces = review_candidates(
        _record([_add_substitute_candidate(), _delete_candidate()]),
        client,
        Path(__file__).parents[1] / "prompts",
        "v5",
        0.0,
        42,
        512,
    )

    assert [call["stage"] for call in backend.calls] == [
        "select_add_substitute",
        "select_delete",
    ]
    assert [
        item["negative_id"]
        for item in json.loads(backend.calls[0]["user_prompt"])["candidates"]
    ] == ["cirr_val_1_attribute_substitute_text"]
    assert [
        item["negative_id"]
        for item in json.loads(backend.calls[1]["user_prompt"])["candidates"]
    ] == ["cirr_val_1_attribute_delete_text"]
    assert "Do not require a DELETE candidate to contradict the" in backend.calls[1][
        "system_prompt"
    ]
    assert all(call["images"] == ["target.png"] for call in backend.calls)
    assert len(traces) == 2
    assert [item.status for item in reviewed] == ["accepted", "accepted"]
    assert [item.edit_operation.value for item in reviewed] == ["substitute", "delete"]


def test_unknown_or_missing_ids_are_retried_fail_closed(tmp_path: Path) -> None:
    invalid = {
        "selected": [{"negative_id": "invented", "reason": "bad id"}],
        "rejected": [],
    }
    valid = {
        "selected": [],
        "rejected": [
            {
                "negative_id": "cirr_val_1_attribute_substitute_text",
                "reason": "The target color is unclear.",
            }
        ],
    }
    backend = FakeVLMBackend([invalid, valid])
    client = StructuredVLMClient(backend, tmp_path, prompt_version="v5", max_retries=1)
    reviewed, traces = review_candidates(
        _record(),
        client,
        Path(__file__).parents[1] / "prompts",
        "v5",
        0.0,
        42,
        512,
    )
    assert len(traces) == 2
    assert reviewed[0].status == "rejected"
    assert len(backend.calls) == 2
