from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cir_qdiag.config import PipelineConfig
from cir_qdiag.models.base import FatalProviderError
from cir_qdiag.models.client import StructuredVLMClient
from cir_qdiag.models.fake import FakeVLMBackend
from cir_qdiag.pipeline import generate_stage, prepare_source, validate_stage
from cir_qdiag.schemas import PinPointGroupRecord
from cir_qdiag.utils.jsonl import read_models
from cir_qdiag.validation import select_validation_positives


PROMPTS = Path(__file__).parents[1] / "prompts"


def _config(tmp_path: Path) -> PipelineConfig:
    rows = []
    for query_id, text in (
        ("query_00001", "Show a red dog running"),
        ("query_00002", "Find a red dog running"),
    ):
        rows.append(
            {
                "query_id": query_id,
                "query_image_signature": "ref",
                "query_image_signature2": "None",
                "instruction": text,
                "positive_candidates": [f"pos{index}" for index in range(10)],
                "negative_candidates": ["offneg"],
                "token_count": 5,
                "length_category": "short",
                "is_question": False,
                "is_imperative": True,
                "query_category": "Swap",
                "l1_interest": "Animals",
            }
        )
    parquet = tmp_path / "pinpoint.parquet"
    pd.DataFrame(rows).to_parquet(parquet)
    return PipelineConfig.model_validate(
        {
            "data": {
                "parquet_path": parquet,
                "image_root": tmp_path / "images",
            },
            "sampling": {"limit": 2, "strategy": "first"},
            "model": {"max_retries": 0},
            "runtime": {
                "output_dir": tmp_path / "out",
                "cache_dir": tmp_path / "cache",
                "prompt_dir": PROMPTS,
                "prompt_version": "v7",
                "max_workers": 1,
            },
        }
    )


def _responses(source: str) -> list[dict]:
    return [
        {
            "object": {
                "negative_text": f"{source}, with a ball",
                "before_span": None,
                "after_span": ", with a ball",
            },
            "attribute": None,
            "predicate": None,
        },
        {
            "object": None,
            "attribute": {
                "negative_text": source.replace("red ", ""),
                "before_span": "red",
                "after_span": None,
            },
            "predicate": {
                "negative_text": source,
                "before_span": None,
                "after_span": None,
            },
        },
        {
            "object": None,
            "attribute": None,
            "predicate": {
                "negative_text": source.replace("running", "sleeping"),
                "before_span": "running",
                "after_span": "sleeping",
            },
        },
    ]


def test_each_instruction_gets_three_text_calls_and_group_gets_one_review(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    source_path = tmp_path / "groups.jsonl"
    candidates_path = tmp_path / "candidates.jsonl"
    final_path = tmp_path / "final.jsonl"
    prepare_source(config, source_path)
    group = list(read_models(source_path, PinPointGroupRecord))[0]

    backend = FakeVLMBackend(
        [
            *_responses(group.instructions[0].text),
            *_responses(group.instructions[1].text),
        ]
    )
    client = StructuredVLMClient(backend, config.runtime.cache_dir, "v7", max_retries=0)
    generate_stage(config, source_path, candidates_path, client, force=True)
    generated = list(read_models(candidates_path, PinPointGroupRecord))[0]

    assert len(generated.candidates) == 8
    assert sum(item.text == item.source_text for item in generated.candidates) == 2
    assert any(item.after_span == ", with a ball" for item in generated.candidates)
    assert [call["stage"] for call in backend.calls] == [
        "generate_add",
        "generate_delete",
        "generate_substitute",
        "generate_add",
        "generate_delete",
        "generate_substitute",
    ]
    assert all(call["images"] == [] for call in backend.calls)

    backend.responses.append(
        {
            "verdicts": [
                {
                    "negative_id": item.negative_id,
                    "decision": (
                        "reject" if item.text == item.source_text else "accept"
                    ),
                    "failure_code": (
                        "not_single_edit"
                        if item.text == item.source_text
                        else None
                    ),
                    "reason": (
                        "The text is unchanged."
                        if item.text == item.source_text
                        else "Valid group hard negative."
                    ),
                }
                for item in generated.candidates
            ]
        }
    )
    validate_stage(config, candidates_path, final_path, client, force=True)
    final = list(read_models(final_path, PinPointGroupRecord))[0]

    assert len(final.negatives) == 6
    selected_positives = select_validation_positives(group, 8)
    assert final.metadata.validated_positive_image_ids == [
        item.image_id for item in selected_positives
    ]
    review = backend.calls[-1]
    assert review["stage"] == "validate_group"
    assert review["images"] == [
        group.references[0].image_path,
        *[item.image_path for item in selected_positives],
    ]
    assert len(review["images"]) == 9
    payload = json.loads(review["user_prompt"])
    assert len(payload["candidates"]) == 8
    assert payload["image_layout"] == {
        "reference_images": [1],
        "positive_images": list(range(2, 10)),
    }
    assert "offneg" not in json.dumps(payload)
    assert all("offneg" not in path for path in review["images"])


class DailyQuotaFailure(RuntimeError):
    status_code = 403
    body = {
        "code": "DAILY_LIMIT_EXCEEDED",
        "message": "您当前每日订阅额度不足，请手动重置额度。",
    }


def test_generation_quota_failure_stops_without_writing_failed_group(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    source_path = tmp_path / "groups.jsonl"
    candidates_path = tmp_path / "candidates.jsonl"
    prepare_source(config, source_path)
    backend = FakeVLMBackend([DailyQuotaFailure("quota exhausted")])
    client = StructuredVLMClient(backend, config.runtime.cache_dir, "v7", max_retries=0)

    with pytest.raises(FatalProviderError, match="provider account or quota"):
        generate_stage(config, source_path, candidates_path, client, force=True)

    assert list(read_models(candidates_path, PinPointGroupRecord)) == []
    assert len(backend.calls) == 1


def test_validation_quota_failure_stops_without_writing_failed_group(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    source_path = tmp_path / "groups.jsonl"
    candidates_path = tmp_path / "candidates.jsonl"
    final_path = tmp_path / "final.jsonl"
    prepare_source(config, source_path)
    group = list(read_models(source_path, PinPointGroupRecord))[0]
    backend = FakeVLMBackend(
        [
            *_responses(group.instructions[0].text),
            *_responses(group.instructions[1].text),
        ]
    )
    client = StructuredVLMClient(backend, config.runtime.cache_dir, "v7", max_retries=0)
    generate_stage(config, source_path, candidates_path, client, force=True)
    backend.responses.append(DailyQuotaFailure("quota exhausted"))

    with pytest.raises(FatalProviderError, match="provider account or quota"):
        validate_stage(config, candidates_path, final_path, client, force=True)

    assert list(read_models(final_path, PinPointGroupRecord)) == []
