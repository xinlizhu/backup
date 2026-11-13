from pathlib import Path

from cir_qdiag.config import PipelineConfig
from cir_qdiag.models.client import StructuredVLMClient
from cir_qdiag.models.fake import FakeVLMBackend
from cir_qdiag.pipeline import (
    _build_fingerprint,
    generate_stage,
    screen_source_stage,
    validate_stage,
)
from cir_qdiag.schemas import ImageRef, SampleRecord
from cir_qdiag.utils.jsonl import read_models, write_jsonl_atomic


def _config(tmp_path: Path) -> PipelineConfig:
    annotation = tmp_path / "cap.json"
    split = tmp_path / "split.json"
    annotation.write_text("[]", encoding="utf-8")
    split.write_text("{}", encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    return PipelineConfig.model_validate(
        {
            "data": {
                "annotation_path": annotation,
                "image_split_path": split,
                "image_root": tmp_path,
            },
            "model": {"name_or_path": model, "max_retries": 0},
            "runtime": {
                "output_dir": tmp_path / "out",
                "cache_dir": tmp_path / "cache",
                "prompt_dir": Path(__file__).parents[1] / "prompts",
                "prompt_version": "v5",
                "max_workers": 1,
            },
        }
    )


def _record(config: PipelineConfig) -> SampleRecord:
    record = SampleRecord(
        sample_id="cirr_val_14178",
        pair_id=14178,
        reference=ImageRef(image_id="ref", image_path="ref.png"),
        target=ImageRef(image_id="target", image_path="target.png"),
        positive_text="Show the empty syringes against a gray background.",
    )
    record.metadata.stage = "source"
    record.metadata.operational_status = "ok"
    record.metadata.build_fingerprint = _build_fingerprint(config)
    return record


def _empty() -> dict:
    return {"object": None, "attribute": None, "predicate": None}


def test_pipeline_is_one_screen_six_generation_and_two_reviews(tmp_path: Path) -> None:
    config = _config(tmp_path)
    source_path = tmp_path / "source.jsonl"
    screened_path = tmp_path / "screened.jsonl"
    rejected_path = tmp_path / "source_rejections.jsonl"
    candidates_path = tmp_path / "candidates.jsonl"
    final_path = tmp_path / "final.jsonl"
    write_jsonl_atomic(source_path, [_record(config)])
    first = {
        "object": {
            "negative_text": "Show the empty syringes and a bottle against a gray background.",
            "before_span": None,
            "after_span": "a bottle",
            "change_summary": "add a bottle",
            "visual_basis": None,
        },
        "attribute": None,
        "predicate": None,
    }
    second = {
        "object": None,
        "attribute": {
            "negative_text": "Show the empty syringes.",
            "before_span": "against a gray background",
            "after_span": None,
            "change_summary": "remove the gray background",
            "visual_basis": None,
        },
        "predicate": None,
    }
    sixth = {
        "object": None,
        "attribute": {
            "negative_text": "Show the full syringes against a gray background.",
            "before_span": "empty",
            "after_span": "full",
            "change_summary": "empty to full",
            "visual_basis": "The Target shows empty rather than full syringes.",
        },
        "predicate": None,
    }
    add_substitute_review = {
        "selected": [
            {
                "negative_id": "cirr_val_14178_attribute_substitute_vision",
                "reason": "Full contradicts the visibly empty target syringes.",
            }
        ],
        "rejected": [
            {
                "negative_id": "cirr_val_14178_object_add_text",
                "reason": "The added bottle is not a sufficiently close hard negative.",
            }
        ],
    }
    delete_review = {
        "selected": [
            {
                "negative_id": "cirr_val_14178_attribute_delete_text",
                "reason": "It removes the gray-background constraint and is strictly weaker.",
            }
        ],
        "rejected": [],
    }
    backend = FakeVLMBackend(
        [
            {"decision": "keep", "reason": "The target satisfies the syringe query."},
            first,
            second,
            _empty(),
            _empty(),
            _empty(),
            sixth,
            add_substitute_review,
            delete_review,
        ]
    )
    client = StructuredVLMClient(
        backend, config.runtime.cache_dir, prompt_version="v5", max_retries=0
    )

    screen_source_stage(config, source_path, screened_path, rejected_path, client)
    assert list(read_models(rejected_path, SampleRecord)) == []
    generate_stage(config, screened_path, candidates_path, client, force=True)
    generated = list(read_models(candidates_path, SampleRecord))[0]
    assert len(generated.candidates) == 3
    assert [call["stage"] for call in backend.calls] == [
        "screen_cirr",
        "generate_text_add",
        "generate_text_delete",
        "generate_text_substitute",
        "generate_vision_add",
        "generate_vision_delete",
        "generate_vision_substitute",
    ]
    assert backend.calls[0]["images"] == ["ref.png", "target.png"]
    assert all(call["images"] == [] for call in backend.calls[1:4])
    assert all(
        call["images"] == ["target.png"] for call in backend.calls[4:7]
    )

    validate_stage(config, candidates_path, final_path, client, force=True)
    final = list(read_models(final_path, SampleRecord))[0]
    assert len(backend.calls) == 9
    assert [call["stage"] for call in backend.calls[-2:]] == [
        "select_add_substitute",
        "select_delete",
    ]
    assert all(call["images"] == ["target.png"] for call in backend.calls[-2:])
    assert len(final.negatives) == 2
    assert {item.negative_id for item in final.negatives} == {
        "cirr_val_14178_attribute_substitute_vision",
        "cirr_val_14178_attribute_delete_text",
    }
    assert not any(
        call["stage"]
        in {
            "parse_positive",
            "realize_negative",
            "audit_visual_context",
            "validate_text_edit",
            "validate_visual_edit",
            "final_candidate_audit",
        }
        for call in backend.calls
    )


def test_source_screen_drops_fundamentally_invalid_triplet(tmp_path: Path) -> None:
    config = _config(tmp_path)
    source_path = tmp_path / "source.jsonl"
    screened_path = tmp_path / "screened.jsonl"
    rejected_path = tmp_path / "source_rejections.jsonl"
    write_jsonl_atomic(source_path, [_record(config)])
    backend = FakeVLMBackend(
        [
            {
                "decision": "drop",
                "reason": "The core object asserted by the query is absent from the target.",
            }
        ]
    )
    client = StructuredVLMClient(
        backend, config.runtime.cache_dir, prompt_version="v5", max_retries=0
    )

    screen_source_stage(config, source_path, screened_path, rejected_path, client)

    assert list(read_models(screened_path, SampleRecord)) == []
    rejected = list(read_models(rejected_path, SampleRecord))
    assert len(rejected) == 1
    assert rejected[0].metadata.stage == "screened"
    assert rejected[0].metadata.source_screen_decision == "drop"
    assert rejected[0].metadata.semantic_status == "rejected"
    assert [call["stage"] for call in backend.calls] == ["screen_cirr"]


def test_source_screen_isolates_one_operational_failure(tmp_path: Path) -> None:
    config = _config(tmp_path)
    source_path = tmp_path / "source.jsonl"
    screened_path = tmp_path / "screened.jsonl"
    rejected_path = tmp_path / "source_rejections.jsonl"
    failed_path = tmp_path / "source_failures.jsonl"
    first = _record(config)
    second = _record(config).model_copy(deep=True)
    second.sample_id = "cirr_val_14179"
    second.pair_id = 14179
    write_jsonl_atomic(source_path, [first, second])
    backend = FakeVLMBackend(
        [
            RuntimeError("Input length exceeds model maximum context length"),
            {"decision": "keep", "reason": "The second sample remains valid."},
        ]
    )
    client = StructuredVLMClient(
        backend, config.runtime.cache_dir, prompt_version="v5", max_retries=0
    )

    screen_source_stage(
        config,
        source_path,
        screened_path,
        rejected_path,
        client,
        failed_path=failed_path,
    )

    kept = list(read_models(screened_path, SampleRecord))
    assert [record.sample_id for record in kept] == ["cirr_val_14179"]
    assert list(read_models(rejected_path, SampleRecord)) == []
    failures = list(read_models(failed_path, SampleRecord))
    assert len(failures) == 1
    assert failures[0].metadata.operational_status == "failed"
    assert failures[0].metadata.source_screen_decision == "pending"
    assert "Input length exceeds" in failures[0].metadata.errors[0]


def test_no_candidates_skips_batch_review(tmp_path: Path) -> None:
    config = _config(tmp_path)
    record = _record(config)
    record.metadata.stage = "candidates"
    record.metadata.num_generated_candidates = 0
    input_path = tmp_path / "candidates.jsonl"
    output_path = tmp_path / "final.jsonl"
    write_jsonl_atomic(input_path, [record])
    backend = FakeVLMBackend([])
    client = StructuredVLMClient(
        backend, config.runtime.cache_dir, prompt_version="v5", max_retries=0
    )
    validate_stage(config, input_path, output_path, client, force=True)
    final = list(read_models(output_path, SampleRecord))[0]
    assert backend.calls == []
    assert final.negatives == []
    assert final.metadata.semantic_status == "rejected"
