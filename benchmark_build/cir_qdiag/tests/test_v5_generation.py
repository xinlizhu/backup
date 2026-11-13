from pathlib import Path

from cir_qdiag.generation.direct import generate_operation
from cir_qdiag.models.client import StructuredVLMClient
from cir_qdiag.models.fake import FakeVLMBackend
from cir_qdiag.schemas import ConstructionMode, EditOperation, ImageRef, SampleRecord


def _record() -> SampleRecord:
    return SampleRecord(
        sample_id="cirr_val_1",
        pair_id=1,
        reference=ImageRef(image_id="r", image_path="ref.png"),
        target=ImageRef(image_id="t", image_path="target.png"),
        positive_text="Show a red cat running beside a tree",
    )


def _response() -> dict:
    return {
        "object": {
            "negative_text": "Show a red dog running beside a tree",
            "before_span": "cat",
            "after_span": "dog",
            "change_summary": "cat to dog",
            "visual_basis": "The Target shows a cat rather than a dog.",
        },
        "attribute": None,
        "predicate": None,
    }


def test_direct_generation_returns_complete_caption_without_realize_call(
    tmp_path: Path,
) -> None:
    backend = FakeVLMBackend([_response()])
    client = StructuredVLMClient(backend, tmp_path, prompt_version="v5", max_retries=0)
    candidates, _ = generate_operation(
        _record(),
        ConstructionMode.VISION_GROUNDED,
        EditOperation.SUBSTITUTE,
        client,
        Path(__file__).parents[1] / "prompts",
        "v5",
        0.0,
        42,
        512,
    )
    assert len(candidates) == 1
    assert candidates[0].text == "Show a red dog running beside a tree"
    assert backend.calls[0]["stage"] == "generate_vision_substitute"
    assert backend.calls[0]["images"] == ["target.png"]
    assert "You do not see the Reference image" in backend.calls[0]["system_prompt"]
    assert len(backend.calls) == 1


def test_text_generation_receives_no_images_and_clears_visual_basis(
    tmp_path: Path,
) -> None:
    backend = FakeVLMBackend([_response()])
    client = StructuredVLMClient(backend, tmp_path, prompt_version="v5", max_retries=0)
    candidates, _ = generate_operation(
        _record(),
        ConstructionMode.TEXT_ONLY,
        EditOperation.SUBSTITUTE,
        client,
        Path(__file__).parents[1] / "prompts",
        "v5",
        0.0,
        42,
        512,
    )
    assert backend.calls[0]["images"] == []
    assert candidates[0].visual_basis is None


def test_null_slots_and_identical_text_do_not_create_candidates(tmp_path: Path) -> None:
    response = {
        "object": {
            "negative_text": _record().positive_text,
            "before_span": None,
            "after_span": "a dog",
            "change_summary": "no change",
            "visual_basis": None,
        },
        "attribute": None,
        "predicate": None,
    }
    backend = FakeVLMBackend([response])
    client = StructuredVLMClient(backend, tmp_path, prompt_version="v5", max_retries=0)
    candidates, _ = generate_operation(
        _record(),
        ConstructionMode.TEXT_ONLY,
        EditOperation.ADD,
        client,
        Path(__file__).parents[1] / "prompts",
        "v5",
        0.0,
        42,
        512,
    )
    assert candidates == []


def test_implicit_target_value_cannot_be_used_as_substitution_anchor(
    tmp_path: Path,
) -> None:
    response = {
        "object": None,
        "attribute": {
            "negative_text": "Show a blue cat running beside a tree",
            "before_span": "yellow",
            "after_span": "blue",
            "change_summary": "yellow to blue",
            "visual_basis": "The Target cat is red, not blue.",
        },
        "predicate": None,
    }
    backend = FakeVLMBackend([response])
    client = StructuredVLMClient(backend, tmp_path, prompt_version="v5", max_retries=0)
    candidates, _ = generate_operation(
        _record(),
        ConstructionMode.VISION_GROUNDED,
        EditOperation.SUBSTITUTE,
        client,
        Path(__file__).parents[1] / "prompts",
        "v5",
        0.0,
        42,
        512,
    )
    assert candidates == []


def test_add_prompts_use_v8_entity_presence_contract() -> None:
    prompt_root = Path(__file__).parents[1] / "prompts"
    for name in ("generate_text_add_v5.txt", "generate_vision_add_v5.txt"):
        prompt = (prompt_root / name).read_text(encoding="utf-8")
        assert "entity identity/presence constraint" in prompt
        assert "with a bird" in prompt
        assert "bare entity" not in prompt

    review_prompt = (prompt_root / "select_add_substitute_v5.txt").read_text(
        encoding="utf-8"
    )
    assert "allow minimal grammatical attachment" in review_prompt
    assert '"and a lamp"' in review_prompt
