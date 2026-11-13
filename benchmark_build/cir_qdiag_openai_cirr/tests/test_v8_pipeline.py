from __future__ import annotations

import json
from pathlib import Path

import pytest

from cir_qdiag_openai_cirr.config import PipelineConfig
from cir_qdiag_openai_cirr.models.base import FatalProviderError
from cir_qdiag_openai_cirr.models.client import StructuredLLMClient
from cir_qdiag_openai_cirr.models.fake import FakeLLMBackend
from cir_qdiag_openai_cirr.pipeline import (
    generate_stage,
    prepare_source,
    validate_stage,
)
from cir_qdiag_openai_cirr.schemas import CIRRSampleRecord
from cir_qdiag_openai_cirr.utils.jsonl import read_models


PROMPTS = Path(__file__).parents[1] / "prompts"


def _config(tmp_path: Path) -> PipelineConfig:
    root = tmp_path / "cirr"
    (root / "train").mkdir(parents=True)
    for image_id in ("r0", "t0", "d0"):
        (root / f"train/{image_id}.png").write_bytes(b"png")
    annotation = tmp_path / "cap.rc2.train.json"
    split = tmp_path / "split.rc2.train.json"
    annotation.write_text(
        json.dumps(
            [
                {
                    "pairid": 0,
                    "reference": "r0",
                    "target_hard": "t0",
                    "caption": "Make the red dog run",
                    "img_set": {"id": 1, "members": ["r0", "t0", "d0"]},
                }
            ]
        ),
        encoding="utf-8",
    )
    split.write_text(
        json.dumps(
            {
                image_id: f"./train/{image_id}.png"
                for image_id in ("r0", "t0", "d0")
            }
        ),
        encoding="utf-8",
    )
    return PipelineConfig.model_validate(
        {
            "data": {
                "annotation_path": annotation,
                "image_split_path": split,
                "image_root": root,
            },
            "sampling": {"limit": 1, "strategy": "first"},
            "model": {"max_retries": 0},
            "runtime": {
                "output_dir": tmp_path / "out",
                "cache_dir": tmp_path / "cache",
                "prompt_dir": PROMPTS,
                "prompt_version": "v8",
                "max_workers": 1,
            },
        }
    )


def _generation_responses() -> list[dict]:
    return [
        {
            "object": {
                "negative_text": "Make the red dog run with a ball",
                "before_span": None,
                "after_span": "with a ball",
            },
            "attribute": None,
            "predicate": None,
        },
        {
            "object": None,
            "attribute": {
                "negative_text": "Make the dog run",
                "before_span": "red ",
                "after_span": None,
            },
            "predicate": None,
        },
        {
            "object": None,
            "attribute": None,
            "predicate": {
                "negative_text": "Make the red dog sleep",
                "before_span": "run",
                "after_span": "sleep",
            },
        },
    ]


def test_sample_gets_three_text_generations_and_one_visual_review(tmp_path: Path) -> None:
    config = _config(tmp_path)
    source_path = tmp_path / "source.jsonl"
    candidates_path = tmp_path / "candidates.jsonl"
    final_path = tmp_path / "final.jsonl"
    prepare_source(config, source_path)

    backend = FakeLLMBackend(_generation_responses())
    client = StructuredLLMClient(backend, config.runtime.cache_dir, "v8", max_retries=0)
    generate_stage(config, source_path, candidates_path, client, force=True)
    generated = list(read_models(candidates_path, CIRRSampleRecord))[0]

    assert len(generated.candidates) == 3
    assert [call["stage"] for call in backend.calls] == [
        "generate_add",
        "generate_delete",
        "generate_substitute",
    ]
    assert all(call["images"] == [] for call in backend.calls)
    assert all(
        item.construction_mode == "text_only_editable"
        for item in generated.candidates
    )

    backend.responses.append(
        {
            "verdicts": [
                {
                    "negative_id": item.negative_id,
                    "decision": "accept",
                    "failure_code": None,
                    "reason": "Valid visually contradicted single-edit counterfactual.",
                }
                for item in generated.candidates
            ]
        }
    )
    validate_stage(config, candidates_path, final_path, client, force=True)
    final = list(read_models(final_path, CIRRSampleRecord))[0]

    assert len(final.negatives) == 3
    review = backend.calls[-1]
    assert review["stage"] == "validate_visual"
    assert review["images"] == [
        str(tmp_path / "cirr/train/r0.png"),
        str(tmp_path / "cirr/train/t0.png"),
    ]
    payload = json.loads(review["user_prompt"])
    assert payload["source_instruction"] == "Make the red dog run"
    assert payload["image_layout"] == {
        "reference_image": 1,
        "target_positive_image": 2,
    }
    assert final.metadata.validation_mode == "reference_target_visual"


class DailyQuotaFailure(RuntimeError):
    status_code = 403
    body = {
        "code": "DAILY_LIMIT_EXCEEDED",
        "message": "daily subscription quota exhausted",
    }


def test_provider_quota_failure_stops_without_writing_failed_sample(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    source_path = tmp_path / "source.jsonl"
    candidates_path = tmp_path / "candidates.jsonl"
    prepare_source(config, source_path)
    backend = FakeLLMBackend([DailyQuotaFailure("quota exhausted")])
    client = StructuredLLMClient(backend, config.runtime.cache_dir, "v8", max_retries=0)

    with pytest.raises(FatalProviderError, match="provider account or quota"):
        generate_stage(config, source_path, candidates_path, client, force=True)

    assert list(read_models(candidates_path, CIRRSampleRecord)) == []
